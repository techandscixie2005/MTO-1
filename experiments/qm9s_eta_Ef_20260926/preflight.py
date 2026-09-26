import copy,hashlib,json,pathlib
import numpy as np
import torch
from e3nn import o3
from dataset import Data,ROOT
from model_factory import build
from objective import losses
from trainer import setup,state_hash,atomic_json,atomic_save
from models_ea import losses as old_losses
def main():
    setup(11);data=Data();cfgs=[json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in ('mto_eta0','mto_eta01','mto_eta1')]
    models=[build(c,data.stats).cuda() for c in cfgs]
    hashes=[state_hash(m.state_dict()) for m in models];assert len(set(hashes))==1
    order_hashes=[]
    for c in cfgs:
        rng=np.random.default_rng(c['seed']);order_hashes.append([hashlib.sha256(rng.permutation(data.parts['train']).tobytes()).hexdigest() for _ in range(3)])
    assert all(x==order_hashes[0] for x in order_hashes)
    atomic_save(models[0].state_dict(),ROOT/'initial_model.pt')
    atomic_json(dict(model_state_sha256=hashes[0],all_three=hashes,first_three_epoch_order_sha256=order_hashes[0],seed=11,
        total_parameters=sum(p.numel() for p in models[0].parameters()),backbone_parameters=sum(p.numel() for p in models[0].core.parameters())),ROOT/'reports/initialization.json')
    x,y=data.batch(data.parts['train'][:4]);m=models[0].double();xd={k:(v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in x.items()}
    yd={k:(v.double() if v.is_floating_point() else v) for k,v in y.items()}
    pred=m(**xd);old=old_losses(pred,(yd['E'],yd['A']),data.stats)[0];new=losses(pred,yd,data.stats,1.)[0]
    params=list(m.parameters());go=torch.autograd.grad(old,params,retain_graph=True,allow_unused=True);gn=torch.autograd.grad(new,params,allow_unused=True)
    diff=max(float((a-b).abs().max()) for a,b in zip(go,gn) if a is not None)
    scale=max(float(a.abs().max()) for a in go if a is not None)
    assert abs(float(new-old))<1e-11 and diff<1e-10*max(scale,1.)
    real_equivalence=dict(dtype='float64',old=float(old),eta1=float(new),loss_abs_diff=abs(float(new-old)),parameter_grad_max_abs_diff=diff,parameter_grad_max_abs=scale)
    # Dense random symmetric PSD targets exercise all tensor directions and nontrivial masks.
    gen=torch.Generator().manual_seed(913)
    e=torch.randn(6,10,generator=gen,dtype=torch.double,requires_grad=True)
    c=torch.randn(6,10,3,3,generator=gen,dtype=torch.double,requires_grad=True)
    a=c@c.transpose(-1,-2);ct=torch.randn(6,10,3,3,generator=gen,dtype=torch.double)
    mask=torch.rand(6,10,generator=gen)>.3
    yt=dict(E=torch.randn(6,10,generator=gen,dtype=torch.double),A=ct@ct.transpose(-1,-2),mask_E=mask,mask_A=mask)
    old=(e[mask]-yt['E'][mask]).square().mean()/data.stats['sE2']+(a[mask]-yt['A'][mask]).square().sum((-2,-1)).mean()/data.stats['sA2']
    new=losses((e,a),yt,data.stats,1.)[0]
    g1=torch.autograd.grad(old,(e,c),retain_graph=True);g2=torch.autograd.grad(new,(e,c),retain_graph=True)
    assert torch.allclose(old,new,atol=1e-12,rtol=1e-12)
    assert all(torch.allclose(u,v,atol=1e-12,rtol=1e-12) for u,v in zip(g1,g2))
    synthetic=dict(old=float(old),eta1=float(new),gradient_max_abs_diff=max(float((u-v).abs().max()) for u,v in zip(g1,g2)))
    # NaN padding is masked before subtraction; printed zeros stay valid.
    masked={k:v.clone() for k,v in yt.items()};masked['E'][~mask]=float('nan');masked['A'][~mask]=float('nan')
    masked_loss=losses((e,a),masked,data.stats,1.)[0];assert torch.allclose(masked_loss,new)
    rot_results=[]
    m.eval()
    with torch.no_grad():
        e0,a0=m(**xd)
        for eta in (0.,.1,1.):
            rows=[]
            for _ in range(3):
                r=o3.rand_matrix(dtype=torch.double,device='cuda')
                xr=dict(xd,pos=xd['pos']@r.T);yr=dict(yd,A=r@yd['A']@r.T)
                er,ar=m(**xr);target_a=r@a0@r.T
                de=float((er-e0).abs().max());da=float((ar-target_a).abs().max())
                dl=abs(float(losses((er,ar),yr,data.stats,eta)[0]-losses((e0,a0),yd,data.stats,eta)[0]))
                mineig=float(torch.linalg.eigvalsh(ar).min())
                assert de<1e-7 and da<1e-7 and dl<1e-7 and mineig>=-1e-10
                rows.append(dict(E_invariance_max_abs=de,A_equivariance_max_abs=da,loss_invariance_abs=dl,minimum_A_eigenvalue=mineig))
            rot_results.append(dict(eta=eta,rotations=rows))
    # Production FP32 path, including gradients and rotation tolerance.
    m=models[1].float();p=m(**x);old=old_losses(p,(y['E'],y['A']),data.stats)[0];new=losses(p,y,data.stats,1.)[0]
    g1=torch.autograd.grad(old,list(m.parameters()),retain_graph=True,allow_unused=True);g2=torch.autograd.grad(new,list(m.parameters()),allow_unused=True)
    gd=max(float((u-v).abs().max()) for u,v in zip(g1,g2) if u is not None)
    gs=max(float(u.abs().max()) for u in g1 if u is not None)
    assert abs(float(old-new))<1e-5 and gd<1e-5*max(1.,gs)
    with torch.no_grad():
        r=o3.rand_matrix(device='cuda');er,ar=m(**dict(x,pos=x['pos']@r.T));e0,a0=m(**x)
        fp32rot=dict(E_max_abs=float((er-e0).abs().max()),A_max_abs=float((ar-r@a0@r.T).abs().max()),minimum_A_eigenvalue=float(torch.linalg.eigvalsh(ar).min()))
        assert fp32rot['E_max_abs']<3e-5 and fp32rot['A_max_abs']<3e-5 and fp32rot['minimum_A_eigenvalue']>-1e-6
    report=dict(passed=True,real_model_eta1_equivalence=real_equivalence,synthetic_masked_equivalence=synthetic,masked_nan_passed=True,
        fp32_eta1=dict(loss_abs_diff=abs(float(old-new)),gradient_max_abs_diff=gd,gradient_max_abs=gs),fp32_rotation=fp32rot,
        rotations=rot_results,identical_initial_parameters=True,identical_shuffle=True,
        notes='Q is computed from final A, never from C. Scratch preflight models discarded; all runs build seed=11 from scratch.')
    atomic_json(report,ROOT/'reports/preflight.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
