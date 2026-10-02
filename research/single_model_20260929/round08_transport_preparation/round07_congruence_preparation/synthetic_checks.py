"""Source-reviewed CPU synthetic algebra/gradients/geometry; zero updates."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import copy,json,math,tempfile,time
from pathlib import Path
import torch
from common import ROOT,sha,read,immutable_json
from congruence import SharedPSDCongruence,element_counts,EPSILON,BOUND,ENERGY_EV_PER_HARTREE
from fresh_model import build_fresh,build_original,base_state,inherited_state,tensor_hash,base_loss
from preflight_helpers import forbid_external_data
from predictor import load_predictor,export_payload
from runtime import atomic_torch

DECISION='637a3cd3006d3f5b84f8ca5314b6213b0bc3017dffeeafb7ed51c2a4c594c60b'


def close(a,b,report,key):
    tolerance=(1e-10,1e-8) if a.dtype==torch.float64 else (2e-6,1e-5)
    assert torch.allclose(a,b,atol=tolerance[0],rtol=tolerance[1]),key
    report[key]=float((a-b).abs().max())


def rejects(call):
    try:call()
    except (AssertionError,ValueError):return True
    raise AssertionError('Expected rejection')


def hand_gate(module):
    with torch.no_grad():
        module.gate[-1].weight.fill_(.005)
        module.gate[-1].bias.fill_(.02)


def rotate(matrix,O):return O@matrix@O.T


def module_checks(dtype):
    torch.manual_seed(20260930)
    C=torch.randn(2,10,3,3,dtype=dtype)*.15
    A=C@C.transpose(-1,-2)
    E=torch.linspace(2,11,20,dtype=dtype).reshape(2,10)
    counts=torch.tensor([[4,1,0,0,0],[2,0,0,1,0]],dtype=dtype)
    modules={mode:SharedPSDCongruence(mode).to(dtype=dtype) for mode in ('original','scalar','tensor')}
    state=modules['tensor'].state_dict()
    for module in modules.values():module.load_state_dict(state)
    report={}
    for mode,module in modules.items():
        out,context=module(E,A,counts)
        assert torch.equal(out,A),mode
        assert sum(p.numel() for p in module.parameters())==177
        assert sum(p.numel() for p in module.parameters() if p.requires_grad)==(0 if mode=='original' else 177)
    report['same_input_identity_bitwise']=True
    # Fixed-B partial derivative and .25 chain factor at identity are distinct.
    for mode in ('scalar','tensor'):
        module=modules[mode]
        energy=E.clone().requires_grad_();matrix=A.clone().requires_grad_()
        out,c=module(energy,matrix,counts);strength=out.diagonal(dim1=-2,dim2=-1).sum(-1)
        derivative=torch.autograd.grad(strength.sum(),c['b'],retain_graph=True)[0]
        s=A.diagonal(dim1=-2,dim2=-1).sum(-1)
        expected=(2*c['q'][:,None]*s if mode=='scalar' else
                  2*(A*c['B'][:,None].transpose(-1,-2)).sum((-2,-1))).sum(1)
        close(derivative,expected,report,mode+'_identity_partial_b_error')
        assert bool((expected>0).all())
        biasgrad=torch.autograd.grad(strength.sum(),module.gate[-1].bias,retain_graph=True)[0]
        close(biasgrad,BOUND*expected.sum().reshape(1),report,mode+'_identity_final_bias_error')
        gradients=torch.autograd.grad(strength.sum(),[energy,matrix,*module.gate.parameters()],allow_unused=True)
        assert torch.equal(gradients[0],torch.zeros_like(energy))
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        assert torch.equal(gradients[2],torch.zeros_like(gradients[2])) and torch.equal(gradients[3],torch.zeros_like(gradients[3]))
        assert gradients[-1].norm()>0 and gradients[-2].norm()>0 and gradients[1].norm()>0
        report[mode+'_identity_gradient_contract']=True
        hand_gate(module)
        energy=E.clone().requires_grad_();matrix=A.clone().requires_grad_()
        out,c=module(energy,matrix,counts);total=out.diagonal(dim1=-2,dim2=-1).sum()
        grads=torch.autograd.grad(total,[energy,matrix,*module.gate.parameters()],retain_graph=True)
        assert all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in grads)
        # Analytical full energy-feature chain with fixed incoming A.
        with torch.no_grad():
            x=c['features'];W1=module.gate[0].weight;z=x@W1.T+module.gate[0].bias
            sig=z.sigmoid();silu_prime=sig*(1+z*(1-sig))
            dgdx=(module.gate[-1].weight*silu_prime)@W1
            rawg=module.gate(x).squeeze(-1)
            dbdx=BOUND*(1-rawg.tanh().square())[:,None]*dgdx
            s=matrix.diagonal(dim1=-2,dim2=-1).sum(-1)
            B=c['B'];b=c['b'];q=c['q']
            if mode=='scalar':dsdb=2*q[:,None]*(1+b*q)[:,None]*s
            else:
                first=(matrix*B[:,None].transpose(-1,-2)).sum((-2,-1))
                second=(matrix*(B@B)[:,None].transpose(-1,-2)).sum((-2,-1))
                dsdb=2*first+2*b[:,None]*second
            weighted=(energy/ENERGY_EV_PER_HARTREE*s).sum(1)
            expectedE=dsdb.sum(1)[:,None]*(dbdx[:,5,None]/(10*ENERGY_EV_PER_HARTREE)
                         +dbdx[:,7,None]*s/(ENERGY_EV_PER_HARTREE*(1+weighted)[:,None]))
        close(grads[0],expectedE,report,mode+'_nonzero_gate_analytic_E_feedback_error')
        report[mode+'_nonzero_gate_E_feedback_l2']=float(grads[0].norm())
        output,c=module(E,A,counts)
        for det in (1,-1):
            O=torch.linalg.qr(torch.randn(3,3,dtype=dtype))[0]
            O=O*torch.linalg.det(O)*det
            actual,rotated=module(E,rotate(A,O),counts)
            close(actual,rotate(output,O),report,f'{mode}_O3_{det}')
            close(rotated['features'],c['features'],report,f'{mode}_features_O3_{det}')
        inv=-torch.eye(3,dtype=dtype)
        close(module(E,rotate(A,inv),counts)[0],output,report,mode+'_inversion')
        V=torch.linalg.qr(torch.randn(2,10,3,3,dtype=dtype))[0]
        gauged=C@V
        close(module(E,gauged@gauged.transpose(-1,-2),counts)[0],output,report,mode+'_factor_gauge')
        perm=torch.tensor([9,1,0,4,7,6,3,2,8,5])
        close(module(E[:,perm],A[:,perm],counts)[0],output[:,perm],report,mode+'_state_permutation')
        L=torch.eye(3,dtype=dtype)+(c['b'][:,None,None]*c['B'] if mode=='tensor' else
                                  (c['b']*c['q'])[:,None,None]*torch.eye(3,dtype=dtype))
        eigen=torch.linalg.eigvalsh(L)
        assert float(eigen.min())>=.75 and float(eigen.max())<=1.25
        assert float(torch.linalg.eigvalsh(output).min())>=-((1e-10) if dtype==torch.float64 else 2e-6)
        report[mode+'_L_eigenvalue_range']=[float(eigen.min()),float(eigen.max())]
        saved_gate=copy.deepcopy(module.gate.state_dict())
        with torch.no_grad():
            module.gate[-1].weight.zero_();module.gate[-1].bias.fill_(-.2)
        suppressed,negative=module(E,A,counts)
        assert bool((negative['b']<0).all())
        assert bool((suppressed.diagonal(dim1=-2,dim2=-1).sum(-1)
                     < A.diagonal(dim1=-2,dim2=-1).sum(-1)).all())
        Ln=torch.eye(3,dtype=dtype)+(negative['b'][:,None,None]*negative['B'] if mode=='tensor'
             else (negative['b']*negative['q'])[:,None,None]*torch.eye(3,dtype=dtype))
        en=torch.linalg.eigvalsh(Ln)
        assert float(en.min())>=.75 and float(en.max())<=1.25
        assert float(torch.linalg.eigvalsh(suppressed).min())>=-(1e-10 if dtype==torch.float64 else 2e-6)
        report[mode+'_negative_b_suppression_PSD_bounds']=True
        module.gate.load_state_dict(saved_gate)
        # Zero/tiny/bright inputs and fixed-input ranks; no normalization tuning.
        fixture=E.new_zeros(2,10,3,3);fixture[0,:,0,0]=1.;fixture[1]=torch.eye(3,dtype=dtype)
        rankout,_=module(E,fixture,counts)
        assert torch.equal(torch.linalg.matrix_rank(fixture),torch.linalg.matrix_rank(rankout))
        for scale in (0.,1e-24,1e4):
            tiny=(A*scale).detach().requires_grad_()
            out,_=module(E,tiny,counts)
            g=torch.autograd.grad(out.sum(),tiny)[0]
            assert bool(torch.isfinite(out).all() & torch.isfinite(g).all())
            if scale==0:assert torch.equal(out,tiny)
        planar=E.new_zeros(2,10,3,3);planar[:,:,2,2]=.2
        out,_=module(E,planar,counts);assert bool((out[:,:,2,2]>0).all())
        mirror=torch.diag(E.new_tensor([1.,1.,-1.]))
        close(module(E,rotate(planar,mirror),counts)[0],out,report,mode+'_planar_normal')
    # Both active gates have identical tensors; isotropic equality is numerical.
    isotropic=torch.eye(3,dtype=dtype).repeat(2,10,1,1)*.125
    close(modules['scalar'](E,isotropic,counts)[0],modules['tensor'](E,isotropic,counts)[0],report,'isotropic_control_candidate')
    _,c=modules['tensor'](E,A,counts)
    close((c['b'][:,None,None]*c['B']).square().sum((-2,-1)),
          3*(c['b']*c['q']).square(),report,'perturbation_Frobenius_norm_squared')
    # Orthogonal rotation of explicit real dipoles in exactly equal-energy block.
    mu=torch.randn(2,10,3,dtype=dtype)*.2;energy=E.clone();energy[:,:3]=4.
    U=torch.linalg.qr(torch.randn(3,3,dtype=dtype))[0]
    changed=mu.clone();changed[:,:3]=U@mu[:,:3]
    a=mu[...,None]*mu[...,None,:];ap=changed[...,None]*changed[...,None,:]
    for mode in ('scalar','tensor'):
        out,c=modules[mode](energy,a,counts);outp,cp=modules[mode](energy,ap,counts)
        close(c['features'],cp['features'],report,mode+'_equal_energy_block_features')
        close(out[:,:3].sum(1),outp[:,:3].sum(1),report,mode+'_equal_energy_block_sum')
        close(out[:,3:],outp[:,3:],report,mode+'_outside_block_unchanged')
        assert not torch.allclose(out[:,:3],outp[:,:3])
    # Target masks do not select gate inputs. Invalid predictions must reject.
    mask=torch.ones(2,10,dtype=torch.bool);mask[0,1]=False
    target={'E':E.clone(),'A':A.clone(),'mask_E':mask,'mask_A':mask}
    target['E'][~mask]=float('nan');target['A'][~mask]=float('nan')
    for mode,module in modules.items():
        energy=E.clone().requires_grad_();matrix=A.clone().requires_grad_()
        out,_=module(energy,matrix,counts)
        loss=base_loss((energy,out),target,{'sE2':.5,'sA2':.1})
        assert all(bool(torch.isfinite(v)) for v in loss.values())
        loss['total'].backward()
        assert bool(torch.isfinite(energy.grad).all() & torch.isfinite(matrix.grad).all())
        bad=E.clone();bad[~mask]=float('nan')
        assert rejects(lambda:module(bad,A,counts))
    report['target_sentinel_masks_but_all_prediction_slots_required']=True
    return report


def geometry():
    z=torch.tensor([6,1,1,1,1,8,1,1]);batch=torch.tensor([0,0,0,0,0,1,1,1])
    pos=torch.tensor([[0,0,0],[.63,.63,.63],[-.63,-.63,.63],[-.63,.63,-.63],[.63,-.63,-.63],
                      [0,0,0],[.76,0,.59],[-.76,0,.59]],dtype=torch.float32)
    edges=[(i,j) for i in range(8) for j in range(8) if i!=j and batch[i]==batch[j]]
    return {'z':z,'pos':pos,'batch':batch,'n':2,'edge_index':torch.tensor(edges).T.contiguous()}


def main():
    start=time.time();assert not (ROOT/'CPU_PREFLIGHT.json').exists()
    review=read(ROOT/'CPU_SOURCE_REVIEW.json')
    assert review['passed'] and review['scope']=='round07_cpu_synthetic_only'
    assert sha(ROOT/'ROUND07_PREPARATION_DECISION.md')==review['root_preparation_decision_sha256']==DECISION
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    torch.set_num_threads(2)
    report={'module_float64':module_checks(torch.float64),'module_float32':module_checks(torch.float32),
            'model_updates':0,'real_geometry_or_targets_read':False,'synthetic_geometry_only':True}
    stats=read(ROOT/'TRAIN_STATISTICS.json');mc=read(ROOT/'model_config.json');x=geometry()
    assert torch.equal(element_counts(x['z'],x['batch'],x['n'],torch.float32),
                       torch.tensor([[4,1,0,0,0],[2,0,0,1,0]],dtype=torch.float32))
    report['hand_derived_CH4_H2O_counts_exact']=True
    assert sha(ROOT/'TRAIN_STATISTICS.json')=='d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae'
    before=torch.get_rng_state().clone()
    with forbid_external_data() as opened:
        base=build_original(mc,stats).eval()
        models={mode:build_fresh(mc,stats,mode).eval() for mode in ('original','scalar','tensor')}
    assert not opened and torch.equal(before,torch.get_rng_state())
    basehash=tensor_hash(base.state_dict());assert basehash=='231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2'
    fullhashes=[tensor_hash(model.state_dict()) for model in models.values()];assert len(set(fullhashes))==1
    gatehashes=[tensor_hash(model.shared_readout.state_dict()) for model in models.values()];assert len(set(gatehashes))==1
    report.update(initial_base_tensor_sha256=basehash,initial_full_tensor_sha256=fullhashes[0],
                  initial_gate_tensor_sha256=gatehashes[0],train_statistics_sha256=sha(ROOT/'TRAIN_STATISTICS.json'))
    with torch.no_grad():reference=base(**x)
    report['full_models']={}
    for mode,model in models.items():
        assert tensor_hash(base_state(model))==basehash
        assert tensor_hash(inherited_state(model))=='3c5d20463a6aa1d4e7c25ee5e57e0ceaa53c17e5a89af35d7761b58e58d661f9'
        assert sum(p.numel() for p in model.shared_readout.parameters())==177
        assert not any(p.requires_grad for p in model.right_adapter.parameters())
        with torch.no_grad():out=model(**x)
        assert all(torch.equal(a,b) for a,b in zip(out,reference))
        mr={'identity_E_A_bitwise':True,'gate_active_parameters':sum(p.numel() for p in model.shared_readout.parameters() if p.requires_grad)}
        if mode!='original':hand_gate(model.shared_readout)
        # Native buffer dtypes remain untouched; only synthetic gate is changed.
        O=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        permutation=torch.tensor([4,2,0,3,1,7,5,6]);inverse=torch.argsort(permutation)
        xp=dict(x,z=x['z'][permutation],pos=x['pos'][permutation]@O.T+torch.tensor([2.,-3.,1.]),
                batch=x['batch'][permutation],edge_index=inverse[x['edge_index']])
        with torch.no_grad():out=model(**x);moved=model(**xp)
        close(moved[0],out[0],mr,'E_reflection_permutation_translation')
        close(moved[1],rotate(out[1],O),mr,'A_reflection_permutation_translation')
        f=lambda p:p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*ENERGY_EV_PER_HARTREE)
        mr['native_f_symmetry_max_abs']=float((f(out)-f(moved)).abs().max())
        with tempfile.TemporaryDirectory(prefix='round07_synthetic_') as directory:
            path=Path(directory)/'synthetic.pt'
            atomic_torch(export_payload(model,mc,stats,{'synthetic_only':True}),path)
            with forbid_external_data(path) as opened:
                loaded=load_predictor(path)
                with torch.no_grad():actual=loaded(**x)
            assert set(opened)=={str(path.resolve())}
            assert all(torch.equal(a,b) for a,b in zip(out,actual))
        mr['one_checkpoint_load_forward_access_and_bitwise_parity']=True
        report['full_models'][mode]=mr
    report.update(source_hashes=review['source_hashes'],source_review_sha256=sha(ROOT/'CPU_SOURCE_REVIEW.json'),
                  root_preparation_decision_sha256=DECISION,seconds=time.time()-start,passed=True)
    immutable_json(report,ROOT/'CPU_PREFLIGHT.json');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
