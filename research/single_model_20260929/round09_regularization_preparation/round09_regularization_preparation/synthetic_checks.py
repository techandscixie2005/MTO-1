"""Fixed CPU-only roster/optimizer/geometry checks, zero full-model updates."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import copy,hashlib,json,tempfile,time
from pathlib import Path
import torch
from common import ROOT,read,sha,immutable_json
from fresh_model import build_fresh,build_original,base_state,tensor_hash,base_loss
from runtime import optimizer,atomic_torch
from predictor import export_payload,load_predictor
from preflight_helpers import forbid_external_data
from regularization import ARMS,roster,optimizer_membership,postclip_diagnostics
DECISION='6b10b8b5a7d75e6a5c46537f8c8b9d126dcc2694d7cfec687e31f120c038d139'
BASE='231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2'
FULL='f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3'


def close(a,b):
    assert torch.allclose(a,b,atol=1e-10 if a.dtype==torch.float64 else 2e-6,
                          rtol=1e-8 if a.dtype==torch.float64 else 1e-5)
    return float((a-b).abs().max())


def optimizer_checks():
    report={};updates=0
    for decay in (0.,1e-4):
        p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
        opt=torch.optim.Adam([p],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=decay,amsgrad=True)
        expected=p.detach().clone();m=torch.zeros_like(p);v=torch.zeros_like(p);vmax=torch.zeros_like(p);errors=[]
        for step,given in enumerate(([20.,-40.],[100.,0.]),1):
            raw=torch.tensor(given,dtype=torch.float64);p.grad=raw.clone()
            norm=float(torch.nn.utils.clip_grad_norm_([p],5.,error_if_nonfinite=True));assert norm>5
            clipped=raw*min(1.,5./(float(raw.norm())+1e-6));close(p.grad,clipped)
            effective=clipped+decay*expected
            if decay:
                pre=raw+decay*expected;pre=pre*min(1.,5./(float(pre.norm())+1e-6))
                assert float((effective-pre).abs().max())>1e-8
            previous_max=vmax.clone()
            m=.9*m+.1*effective;v=.999*v+.001*effective.square();vmax=torch.maximum(vmax,v)
            if step==2:assert v[1]<previous_max[1] and vmax[1]==previous_max[1]
            updated=expected-(.001/(1-.9**step))*m/(vmax.sqrt()/((1-.999**step)**.5)+1e-8)
            if decay and step==1:
                g=clipped;wm=.1*g;wv=.001*g.square()
                adamw=expected*(1-.001*decay)-(.001/(1-.9))*wm/(wv.sqrt()/((1-.999)**.5)+1e-8)
                assert float((updated-adamw).abs().max())>1e-8
            opt.step();updates+=1;st=opt.state[p]
            errors.extend([close(p,updated),close(st['exp_avg'],m),close(st['exp_avg_sq'],v),close(st['max_exp_avg_sq'],vmax)])
            assert int(st['step'])==step;expected=updated
        report[str(decay)]={'steps':2,'max_abs':max(errors),'task_clip_active':True,'AMSGrad_retains_larger_past_second_moment':True}
    skipped=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64));zero=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    opt=torch.optim.Adam([skipped,zero],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,amsgrad=True)
    skipped.grad=None;zero.grad=torch.zeros_like(zero)
    torch.nn.utils.clip_grad_norm_([skipped,zero],5.,error_if_nonfinite=True);opt.step();updates+=1
    assert float(skipped)==2. and skipped not in opt.state and float(zero)<2. and int(opt.state[zero]['step'])==1
    report.update(grad_none_skips_decay_and_state=True,zero_gradient_receives_decay=True,toy_optimizer_step_calls=updates)
    assert updates==5
    return report


def geometry():
    z=torch.tensor([6,1,1,1,1,8,1,1]);batch=torch.tensor([0,0,0,0,0,1,1,1])
    pos=torch.tensor([[0,0,0],[.63,.63,.63],[-.63,-.63,.63],[-.63,.63,-.63],[.63,-.63,-.63],
                      [0,0,0],[.76,0,.59],[-.76,0,.59]],dtype=torch.float32)
    edges=[(i,j) for i in range(8) for j in range(8) if i!=j and batch[i]==batch[j]]
    return {'z':z,'pos':pos,'batch':batch,'n':2,'edge_index':torch.tensor(edges).T.contiguous()}


def mask_checks():
    E=torch.ones(2,10,requires_grad=True);A=torch.eye(3).expand(2,10,3,3).clone().requires_grad_()
    mask=torch.ones(2,10,dtype=torch.bool);mask[0,2]=False
    et=torch.zeros(2,10);at=torch.zeros(2,10,3,3);et[~mask]=float('nan');at[~mask]=float('nan')
    loss=base_loss((E,A),{'E':et,'A':at,'mask_E':mask,'mask_A':mask},{'sE2':.5,'sA2':.1})
    loss['total'].backward();assert torch.isfinite(E.grad).all() and torch.isfinite(A.grad).all()
    assert torch.count_nonzero(E.grad[~mask])==torch.count_nonzero(A.grad[~mask])==0
    assert float(loss['energy'])==2. and torch.isclose(loss['trace'],torch.tensor(30.))
    zero_f=E.detach().double()*torch.zeros_like(A).double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*27.211386245988)
    assert torch.count_nonzero(zero_f)==0
    return {'masked_NaN_target_forward_backward':True,'zero_target_loss_included':True,'zero_native_f_preserved':True}


def main():
    start=time.time();assert not (ROOT/'CPU_PREFLIGHT.json').exists() and not (ROOT/'PARAMETER_ROSTER.json').exists()
    review=read(ROOT/'CPU_SOURCE_REVIEW.json');assert review['passed'] and review['scope']=='round09_cpu_synthetic_only'
    assert sha(ROOT/'ROUND09_PREPARATION_DECISION.md')==review['root_preparation_decision_sha256']==DECISION
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    torch.set_num_threads(2);cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json')
    assert list(cfg['arms'])==list(ARMS);before=torch.get_rng_state().clone()
    with forbid_external_data() as opened:
        original=build_original(mc,stats).eval()
        models={arm:build_fresh(mc,stats,'original').eval() for arm in ARMS}
    assert not opened and torch.equal(before,torch.get_rng_state())
    contracts=[roster(m) for m in models.values()];assert contracts[0]==contracts[1]
    original_roster=[{'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel()}
                     for name,p in original.named_parameters() if p.requires_grad]
    assert contracts[0]['included']==original_roster
    assert tensor_hash(original.state_dict())==BASE
    for model in models.values():assert tensor_hash(base_state(model))==BASE and tensor_hash(model.state_dict())==FULL
    immutable_json({'format':'round09_parameter_roster_v1','frozen_before_any_optimizer_update':True,
        'contract':contracts[0],'unwrapped_original_ordered_roster_equal':True,
        'initial_base_tensor_sha256':BASE,'initial_full_tensor_sha256':FULL,
        'source_review_sha256':sha(ROOT/'CPU_SOURCE_REVIEW.json'),'source_hashes':review['source_hashes']},ROOT/'PARAMETER_ROSTER.json')
    for arm,model in models.items():optimizer_membership(model,optimizer(model,cfg,arm),cfg,arm)
    report={'optimizer_arithmetic':optimizer_checks(),'mask_checks':mask_checks(),'model_updates':0,
            'real_geometry_or_targets_read':False,'synthetic_geometry_only':True,'parameter_roster_sha256':sha(ROOT/'PARAMETER_ROSTER.json')}
    x=geometry();mask=torch.ones(2,10,dtype=torch.bool)
    target={'E':torch.ones(2,10)*5.,'A':torch.eye(3).expand(2,10,3,3)*.1,'mask_E':mask,'mask_A':mask}
    gradients=[];outputs=[];losses=[];diagnostics={}
    for arm,model in models.items():
        pred=model(**x);loss=base_loss(pred,target,stats);loss['total'].backward()
        gradients.append({name:p.grad.detach().clone() if p.grad is not None else None for name,p in model.named_parameters()})
        outputs.append(tuple(t.detach() for t in pred));losses.append(float(loss['total']))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        saved={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
        diagnostics[arm]=postclip_diagnostics(model,cfg['arms'][arm]['weight_decay'])
        assert all(torch.equal(p.grad,saved[name]) for name,p in model.named_parameters() if p.grad is not None)
        assert tensor_hash(model.state_dict())==FULL
    assert losses[0]==losses[1]
    for name in gradients[0]:
        a,b=gradients[0][name],gradients[1][name]
        assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)),name
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    assert diagnostics['zero_decay']['coupled_term_l2']==0 and diagnostics['coupled_l2']['coupled_term_l2']>0
    model=models['coupled_l2'];model.zero_grad(set_to_none=True)
    first=next(p for p in model.parameters() if p.requires_grad);first.grad=torch.zeros_like(first)
    zero=postclip_diagnostics(model,1e-4);assert zero['task_norm_zero']==zero['task_norm_below_floor']==1
    assert zero['parameters_with_grad']==1 and zero['parameters_grad_none']==134
    O=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]]);perm=torch.tensor([4,2,0,3,1,7,5,6]);inv=torch.argsort(perm)
    xp=dict(x,z=x['z'][perm],pos=x['pos'][perm]@O.T+torch.tensor([2.,-3.,1.]),batch=x['batch'][perm],edge_index=inv[x['edge_index']])
    with torch.no_grad():a=model(**x);b=model(**xp)
    symmetry={'E_max_abs':close(a[0],b[0]),'A_max_abs':close(O@a[1]@O.T,b[1])}
    f=lambda p:p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*27.211386245988)
    symmetry['native_f_max_abs']=float((f(a)-f(b)).abs().max())
    with tempfile.TemporaryDirectory(prefix='round09_synthetic_') as directory:
        path=Path(directory)/'synthetic.pt';atomic_torch(export_payload(model,mc,stats,{'synthetic_only':True}),path)
        with forbid_external_data(path) as opened:
            loaded=load_predictor(path)
            with torch.no_grad():actual=loaded(**x)
        assert set(opened)=={str(path.resolve())} and all(torch.equal(u,v) for u,v in zip(a,actual))
    assert all(tensor_hash(m.state_dict())==FULL for m in models.values())
    report.update(initial_base_tensor_sha256=BASE,initial_full_tensor_sha256=FULL,
        initial_transport_tensor_sha256='9b4cfe4de1555a051e4ef745dae7f6121c1a4f09bd35bd41dd2db9816110eceb',
        train_statistics_sha256=sha(ROOT/'TRAIN_STATISTICS.json'),equal_initial_task_loss_and_gradients=True,
        constructor_CPU_RNG_unchanged=True,postclip_diagnostics=diagnostics,zero_gradient_diagnostic=zero,
        full_model_symmetry=symmetry,one_checkpoint_access_and_bitwise_parity=True,
        source_hashes=review['source_hashes'],source_review_sha256=sha(ROOT/'CPU_SOURCE_REVIEW.json'),
        root_preparation_decision_sha256=DECISION,seconds=time.time()-start,passed=True)
    immutable_json(report,ROOT/'CPU_PREFLIGHT.json');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
