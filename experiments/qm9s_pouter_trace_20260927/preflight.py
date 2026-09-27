import copy,hashlib,importlib.util,json,random,time
import numpy as np
import torch
from e3nn import o3
from dataset import Data,ROOT
from model_factory import build,MTOEA
from objective import losses
from trainer import setup,state_hash,atomic_json,atomic_save

def grad(m,x,y,eta):
    m.zero_grad(set_to_none=True);v=losses(m(**x),y,data.stats,eta)[0];v.backward()
    return float(v),{k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
def close(a,b,atol=1e-7,rtol=1e-5):
    assert torch.allclose(a,b,atol=atol,rtol=rtol),float((a-b).abs().max())
def optim(m,c):return torch.optim.Adam(m.parameters(),lr=c['lr'],amsgrad=True,weight_decay=c['weight_decay'])
def scheduler(o,c):return torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=.5,patience=c['plateau_patience'],threshold=1e-4,threshold_mode='rel',min_lr=c['min_lr'])
def step(m,o,x,y,eta):
    o.zero_grad(set_to_none=True);v=losses(m(**x),y,data.stats,eta)[0];v.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);o.step();return float(v)

def main():
    global data
    setup(11);data=Data();cfgs={n:json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in ('G1','G2','G3','G4')}
    report={};x,y=data.batch(data.parts['train'][:4])
    models={n:build(c,data.stats).cuda() for n,c in cfgs.items()}
    hashes={n:state_hash(m.state_dict()) for n,m in models.items()}
    assert hashes['G1']==hashes['G2'] and hashes['G3']==hashes['G4']
    for k,v in models['G1'].state_dict().items():
        if k.startswith(('core.','mto.')):
            assert all(torch.equal(v,m.state_dict()[k]) for m in models.values()),k
    report['initialization_equal']=hashes
    expected=json.loads((ROOT/'reports/epoch_order_hashes.json').read_text())
    for c in cfgs.values():
        rng=np.random.default_rng(c['seed'])
        for row in expected:assert hashlib.sha256(rng.permutation(data.parts['train']).tobytes()).hexdigest()==row['sha256']
    report['all_1000_epoch_orders_equal']=True
    # Independent pinned forward and pinned eta=1 objective.
    spec=importlib.util.spec_from_file_location('pinned_objective',ROOT/'pinned_reference/objective.py');refobj=importlib.util.module_from_spec(spec);spec.loader.exec_module(refobj)
    torch.manual_seed(11);ref=MTOEA(cfgs['G1'],data.stats).cuda()
    assert state_hash(ref.state_dict())==hashes['G1']
    eq={}
    for dtype in (torch.float32,torch.float64):
        m=models['G1'].to(dtype);ref=ref.to(dtype)
        xx={k:v.to(dtype) if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in x.items()}
        yy={k:v.to(dtype) if v.is_floating_point() else v for k,v in y.items()}
        p=m(**xx);r=ref(**xx);tol=2e-5 if dtype==torch.float32 else 1e-10
        for u,v in zip(p,r):close(u,v,tol,tol)
        a=losses(p,yy,data.stats,1)[0];b=refobj.losses(r,yy,data.stats,1)[0]
        ga=torch.autograd.grad(a,tuple(m.parameters()),allow_unused=True);gb=torch.autograd.grad(b,tuple(ref.parameters()),allow_unused=True)
        delta=max(float((u-v).abs().max()) for u,v in zip(ga,gb) if u is not None)
        close(a,b,tol,tol)
        for u,v in zip(ga,gb):
            assert (u is None)==(v is None)
            if u is not None:close(u,v,tol,tol)
        # Also check decomposition against direct full-A Frobenius objective.
        p=m(**xx);direct=(p[0]-yy['E']).square().mean()/data.stats['sE2']+(p[1]-yy['A']).square().sum((-1,-2)).mean()/data.stats['sA2']
        decomposed=losses(p,yy,data.stats,1)[0]
        gd=torch.autograd.grad(direct,tuple(m.parameters()),retain_graph=True,allow_unused=True);gq=torch.autograd.grad(decomposed,tuple(m.parameters()),allow_unused=True)
        tol=2e-5 if dtype==torch.float32 else 1e-10
        for u,v in zip(gd,gq):
            if u is not None:close(u,v,atol=tol,rtol=tol)
        eq[str(dtype)]=dict(forward_max_abs=max(float((u-v).abs().max()) for u,v in zip(p,r)),pinned_gradient_max_abs=delta,full_A_loss_difference=abs(float(direct-decomposed)),absolute_and_relative_tolerance=tol)
    report['G1_reference_equivalence']=eq;models['G1'].float();del ref
    # The trace objective cannot penalize a traceless change, unlike diagonal MSE.
    E=torch.zeros(2,10,device='cuda',requires_grad=True);A=torch.eye(3,device='cuda').expand(2,10,3,3).clone().requires_grad_()
    yy=dict(E=torch.zeros_like(E),A=A.detach().clone(),mask_E=torch.ones_like(E,dtype=torch.bool),mask_A=torch.ones_like(E,dtype=torch.bool))
    yy['A']+=torch.diag(torch.tensor([.5,-.5,0.],device='cuda'))
    assert float(losses((E,A),yy,data.stats,0)[0])==0
    yy['A'].zero_();v=losses((E,A),yy,data.stats,0)[0];close(v,torch.tensor(9/(3*data.stats['sA2']),device='cuda'))
    for eta in (0,1):
        ym={k:v.clone() for k,v in yy.items()};ym['mask_A'][0]=False;ym['mask_E'][0]=False
        before=losses((E,A),ym,data.stats,eta)[0];ym['A'][0]=float('nan');ym['E'][0]=float('nan')
        after=losses((E,A),ym,data.stats,eta)[0];close(before,after)
        gg=torch.autograd.grad(after,(E,A),retain_graph=True,allow_unused=True)
        assert all(torch.isfinite(g).all() for g in gg if g is not None)
        ym['mask_A'].zero_();ym['mask_E'].zero_();assert float(losses((E,A),ym,data.stats,eta)[0])==0
    report['trace_not_diagonal_MSE_mask_NaN_empty_and_valid_zero_labels']=True
    for n in ('G2','G4'):
        m=models[n];v,a=grad(m,x,y,0);yy={k:v.clone() for k,v in y.items()};yy['E'].fill_(float('nan'))
        w,b=grad(m,x,yy,0);assert abs(v-w)<1e-5 and a.keys()==b.keys()
        for k in a:close(a[k],b[k],atol=2e-5,rtol=2e-5)
        frozen={k:p.clone() for k,p in m.named_parameters() if not p.requires_grad}
        assert set(frozen)=={'decoder.energy_offset','decoder.energy_head.weight','decoder.energy_head.bias'}
        o=optim(m,cfgs[n]);step(m,o,x,y,0)
        assert all(torch.equal(v,dict(m.named_parameters())[k]) for k,v in frozen.items())
        report[n+'_energy_label_gradient_independence_and_frozen_head']=dict(passed=True,gradient_max_abs_difference=max(float((a[k]-b[k]).abs().max()) for k in a),frozen_parameters=list(frozen))
    # Fresh P-only model; decoder receives only the two irreps in P.
    m=build(cfgs['G3'],data.stats).cuda().double().eval()
    xx={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in x.items()}
    with torch.no_grad():
        e,a,mu=m(**xx,export=True);assert float(mu.square().sum())>0
        assert not any(('beta' in k or 'tensor_gate' in k or 'cartesian_basis' in k) for k in m.decoder.state_dict())
        assert m.cg.tp.irreps_out==o3.Irreps('16x0e + 16x1o')
        close(a,a.transpose(-1,-2),atol=0,rtol=0);close(a.diagonal(dim1=-2,dim2=-1).sum(-1),mu.square().sum(-1),atol=1e-12,rtol=1e-12)
        eig=torch.linalg.eigvalsh(a);assert float(eig.min())>-1e-11 and float(eig[...,:2].abs().max())<1e-11
        rows=[]
        for r in [o3.rand_matrix(dtype=torch.double,device='cuda') for _ in range(3)]+[-torch.eye(3,dtype=torch.double,device='cuda')]:
            er,ar,mur=m(**dict(xx,pos=xx['pos']@r.T),export=True)
            close(er,e,1e-8,1e-8);close(mur,mu@r.T,1e-8,1e-8);close(ar,r@a@r.T,1e-8,1e-8)
            rows.append(dict(det=float(torch.det(r)),mu_error=float((mur-mu@r.T).abs().max()),A_error=float((ar-r@a@r.T).abs().max())))
        perm=torch.randperm(len(xx['z']),device='cuda');inverse=torch.argsort(perm)
        xp=dict(xx,z=xx['z'][perm],pos=xx['pos'][perm],batch=xx['batch'][perm],edge_index=inverse[xx['edge_index']])
        ep,ap,mup=m(**xp,export=True);close(ep,e,1e-8,1e-8);close(ap,a,1e-8,1e-8);close(mup,mu,1e-8,1e-8)
        captured=[]
        hook=m.decoder.register_forward_pre_hook(lambda module,args:captured.append(args))
        m(**xx);hook.remove();assert len(captured)==1 and len(captured[0])==1 and set(captured[0][0])=={'0e','1o'}
        # Centrosymmetric synthetic octahedron, all atoms identical.
        pos=torch.cat([torch.eye(3),-torch.eye(3)]).to(device='cuda',dtype=torch.double)
        ii,jj=torch.where(~torch.eye(6,dtype=torch.bool,device='cuda'))
        xs=dict(z=torch.ones(6,dtype=torch.long,device='cuda'),pos=pos,batch=torch.zeros(6,dtype=torch.long,device='cuda'),n=1,edge_index=torch.stack([ii,jj]))
        _,ac,muc=m(**xs,export=True);assert float(muc.abs().max())<1e-10
        report['centrosymmetric_geometry']=dict(mu_max_abs=float(muc.abs().max()),A_max_abs=float(ac.abs().max()),limitation='Ordinary polar 1o head forced to zero by inversion; nonzero transition intensity cannot be represented. No samples removed.')
        report['P_only_equivariance']=dict(rotations_and_inversion=rows,permutation_mu_error=float((mup-mu).abs().max()),minimum_eigenvalue=float(eig.min()),rank_at_most_one=True,decoder_only_P=True)
    # FP32 production batch=64 and resumable Adam/AMSGrad, scheduler and RNG.
    x64,y64=data.batch(data.parts['train'][:64]);resume={}
    for n in cfgs:
        c=cfgs[n];m=build(c,data.stats).cuda();o=optim(m,c);s=scheduler(o,c)
        first=step(m,o,x64,y64,c['eta']);s.step(first)
        g=np.random.default_rng(11);order=g.permutation(data.parts['train']);cursor=64
        ck=dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),cursor=cursor,order=order,
            rng_numpy=g.bit_generator.state,rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),rng_python=random.getstate(),rng_numpy_global=np.random.get_state())
        path=ROOT/'reports'/f'scratch_resume_{n}.pt';atomic_save(ck,path)
        xn,yn=data.batch(order[cursor:cursor+64]);continued=step(m,o,xn,yn,c['eta']);s.step(continued)
        target={k:v.clone() for k,v in m.state_dict().items()}
        m2=build(c,data.stats).cuda();o2=optim(m2,c);s2=scheduler(o2,c);ck=torch.load(path,map_location='cuda',weights_only=False)
        m2.load_state_dict(ck['model']);o2.load_state_dict(ck['optimizer']);s2.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_torch'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu());random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy_global']);g.bit_generator.state=ck['rng_numpy']
        xr,yr=data.batch(ck['order'][ck['cursor']:ck['cursor']+64]);restored=step(m2,o2,xr,yr,c['eta']);s2.step(restored)
        delta=max(float((v-m2.state_dict()[k]).abs().max()) for k,v in target.items() if v.numel())
        assert abs(restored-continued)<2e-5*max(1,abs(continued)) and delta<2e-6,(n,delta,continued,restored)
        for k,v in s.state_dict().items():
            if isinstance(v,float):assert v==s2.state_dict()[k] or abs(v-s2.state_dict()[k])<2e-5*max(1,abs(v))
            else:assert v==s2.state_dict()[k]
        resume[n]=dict(batch_size=64,first_loss=first,next_loss=continued,resumed_loss=restored,model_max_abs_difference=delta,optimizer_steps=int(next(iter(o2.state.values()))['step']))
        path.unlink();del m,m2,o,o2
    report['FP32_gradient_and_resume']=resume
    report.update(passed=True,time=time.time(),scratch_training_discarded=True,formal_runs_load_frozen_initial_files=True)
    atomic_json(report,ROOT/'reports/preflight.json');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
