import json,time
import numpy as np
import torch
from data_ea import Data,ROOT
from models_ea import build,losses
from train_ea import setup,atomic_json
from torch_geometric.nn import radius_graph
setup(11);data=Data();stats=data.stats
# Check user-defined global normalization using all training targets and FP64 reductions.
tr=torch.as_tensor(data.parts['train'],device='cuda');E=data.E[tr].double();A=data.A[tr].double()
le=((E-E.mean(0))**2).mean().item()/stats['sE2'];la=A.square().sum((-2,-1)).mean().item()/stats['sA2']
assert abs(le-1)<1e-6 and abs(la-1)<1e-6
del E,A,tr
cfg=json.loads((ROOT/'configs/mto_reference.json').read_text());base=build({**cfg,'model':'detanet'},stats).cuda();mto=build(cfg,stats).cuda()
for k,v in base.core.state_dict().items():assert torch.equal(v,mto.core.state_dict()[k]),k
reports=[]
for name,model in [('detanet',base),('mto',mto),('mto32',build({**cfg,'mto_channels':32},stats).cuda())]:
    x,y=data.batch(data.parts['train'][:4]);p=model(**x)
    assert p[0].shape==(4,10) and p[1].shape==(4,10,3,3)
    edges=radius_graph(x['pos'],r=5.,batch=x['batch'])
    assert set(map(tuple,edges.T.cpu().tolist()))==set(map(tuple,x['edge_index'].T.cpu().tolist()))
    q=torch.linalg.qr(torch.randn(3,3,device='cuda')).Q
    errors={}
    for kind,U in [('rotation',q),('reflection',-q)]:
        xx={**x,'pos':x['pos']@U.T+torch.tensor([1.,2.,3.],device='cuda')}
        pp=model(**xx);expected=U@p[1]@U.T
        eerr=(pp[0]-p[0]).abs().max().item();aerr=(pp[1]-expected).abs().max().item()
        assert torch.allclose(pp[0],p[0],atol=2e-4,rtol=2e-4),(name,kind,eerr)
        assert torch.allclose(pp[1],expected,atol=2e-4,rtol=2e-3),(name,kind,aerr)
        errors[kind]=dict(E_max_abs=eerr,A_max_abs=aerr)
    perm=torch.cat([(x['batch']==b).nonzero().flatten().flip(0) for b in range(x['n'])]);inv=torch.argsort(perm)
    xx={**x,'z':x['z'][perm],'pos':x['pos'][perm],'batch':x['batch'][perm],'edge_index':inv[x['edge_index']]}
    pp=model(**xx);assert torch.allclose(pp[0],p[0],atol=2e-4,rtol=2e-4);assert torch.allclose(pp[1],p[1],atol=2e-4,rtol=2e-3)
    assert torch.linalg.eigvalsh(p[1]).min().item()>-1e-5
    if name.startswith('mto'):
        # New lightweight property path must equal the pinned upstream physical output.
        from models import SpectrumModel
        _,old=SpectrumModel.forward(model,**x,export=True)
        assert torch.allclose(p[0],old['E'],atol=1e-5,rtol=1e-5) and torch.allclose(p[1],old['A'],atol=1e-5,rtol=1e-5)
        errors['upstream_property_path']=dict(E_max_abs=(p[0]-old['E']).abs().max().item(),A_max_abs=(p[1]-old['A']).abs().max().item())
    loss=losses(p,y,stats)[0];loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(p.grad.abs().sum().item() for p in model.core.parameters() if p.grad is not None)>0
    if name.startswith('mto'):
        for module in (model.mto.router,model.cg,model.decoder):assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)>0
    opt=torch.optim.Adam(model.parameters(),lr=.001,amsgrad=True)
    model.zero_grad(set_to_none=True);start=time.monotonic()
    for i in range(3):
        xx,yy=data.batch(data.parts['train'][i*64:(i+1)*64]);opt.zero_grad(set_to_none=True)
        ll=losses(model(**xx),yy,stats)[0];assert torch.isfinite(ll);ll.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
    torch.cuda.synchronize()
    from train_ea import atomic_save
    temp=ROOT/'reports/preflight_checkpoint.pt'
    atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.cuda.get_rng_state()),temp)
    restored=torch.load(temp,map_location='cuda',weights_only=False)
    for k,v in model.state_dict().items():assert torch.equal(v,restored['model'][k])
    opt.load_state_dict(restored['optimizer']);torch.cuda.set_rng_state(restored['rng'].cpu())
    reports.append(dict(name=name,parameters=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for p in model.core.parameters()),
        errors=errors,initial_loss=float(loss.detach()),three_steps_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated()))
atomic_json(dict(status='PASS',normalized_baselines=dict(energy_state_mean=le,zero_A=la),checks=reports,time=time.time()),ROOT/'reports/preflight.json')
print(json.dumps(reports,indent=2))
