"""Synthetic CPU checks only: no molecular data, checkpoint, or optimizer."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import copy,json
import numpy as np
import torch
from common import ROOT,CAMPAIGN,sha,read,immutable_json
from fresh_model import build_fresh,base_state,tensor_hash,base_loss
from objective import loss_components,F_CONST,F_VARIANCE


def rejected(call):
    try:call()
    except (AssertionError,ValueError):return True
    raise AssertionError('Expected rejection')


def main():
    assert not (ROOT/'CPU_PREFLIGHT.json').exists(),'Completed synthetic checks must not repeat'
    torch.set_num_threads(2)
    stats=read(ROOT/'TRAIN_STATISTICS.json');mc=read(ROOT/'model_config.json')
    prior=read(CAMPAIGN/'round05_scratch_preparation/FROZEN_MANIFEST.json')
    assert sha(ROOT/'ROUND06_PREPARATION_DECISION.md')=='d8f7478f77bf660c32e75897d555b13f972e64abaac4385e2c9b44ab3e2e6458'
    assert sha(ROOT/'TRAIN_STATISTICS.json')=='d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae'
    assert sha(CAMPAIGN/'round05_scratch_preparation/FROZEN_MANIFEST.json')=='66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1'
    # Rehash inherited source inputs as opaque bytes; never decode data/weights.
    for path,digest in prior['source_hashes'].items():assert sha(path)==digest
    checks={}
    energy=torch.tensor([[3.,7.,2.],[11.,5.,9.]],requires_grad=True)
    factor=torch.tensor([[[[.3,.1,0],[0,.2,0],[0,0,.1]],[[.001,0,0],[0,.002,0],[0,0,.003]],[[0.,0,0],[0,0,0],[0,0,0]]],
                          [[[1.1,.2,0],[.1,.8,0],[0,0,.7]],[[.5,0,0],[0,.4,0],[0,0,.3]],[[.2,0,0],[0,.1,0],[0,0,.05]]]])
    matrix=(factor@factor.transpose(-1,-2)).requires_grad_()
    mask=torch.ones((2,3),dtype=torch.bool)
    target={'E':torch.tensor([[3.2,6.8,2.1],[10.7,5.1,9.2]]),'A':torch.eye(3).repeat(2,3,1,1)*.12,
            'f':torch.tensor([[0.,.0001,0.],[.8,.04,.013]],dtype=torch.float64),
            'mask_E':mask,'mask_A':mask,'mask_f':mask}
    original=base_loss((energy,matrix),target,stats)
    trace=loss_components((energy,matrix),target,stats,'trace');raw=loss_components((energy,matrix),target,stats,'raw_f')
    checks['original_control_total_energy_trace_bitwise']=all(torch.equal(trace[k],original[k]) for k in ('total','energy','trace'))
    assert checks['original_control_total_energy_trace_bitwise']
    e64=energy.detach().double();a64=matrix.detach().double();f64=target['f'].float().double();s64=a64.diagonal(dim1=-2,dim2=-1).sum(-1)
    c64=2/(3*27.211386245988)
    assert F_CONST==c64
    expected=((c64*e64*s64-f64)**2).mean()/stats['f_variance']
    assert np.isclose(float(raw['raw_f']),float(expected),atol=2e-6,rtol=1e-5)
    checks['FP32_loss_vs_independent_FP64_formula_abs']=abs(float(raw['raw_f'])-float(expected))
    assert trace['raw_f'].dtype==raw['total'].dtype==torch.float32
    assert torch.equal(raw['total'],raw['energy']+raw['raw_f'])
    gE,gA=torch.autograd.grad(raw['raw_f'],(energy,matrix),retain_graph=True)
    residual=c64*e64*s64-f64;N=energy.numel()
    expectedE=2*c64*s64*residual/(N*stats['f_variance'])
    expectedA=(2*c64*e64*residual/(N*stats['f_variance']))[...,None,None]*torch.eye(3,dtype=torch.float64)
    assert torch.allclose(gE.double(),expectedE,atol=2e-6,rtol=1e-5)
    assert torch.allclose(gA.double(),expectedA,atol=2e-6,rtol=1e-5)
    checks['analytic_E_gradient_max_abs']=float((gE.double()-expectedE).abs().max())
    checks['analytic_A_gradient_max_abs']=float((gA.double()-expectedA).abs().max())
    assert gE.norm()>0 and gA.norm()>0
    # Quadratic factor parameterization: zero C cannot acquire a direct
    # strength gradient even when its printed target is nonzero.
    c_factor=factor.detach().clone().requires_grad_();a_factor=c_factor@c_factor.transpose(-1,-2)
    factor_target=copy.deepcopy(target);factor_target['f'][0,2]=.25
    factor_loss=loss_components((energy,a_factor),factor_target,stats,'raw_f')['raw_f']
    gC,gFactorA=torch.autograd.grad(factor_loss,(c_factor,a_factor),retain_graph=True)
    cf64=c_factor.detach().double();sf64=cf64.square().sum((-2,-1))
    rf64=c64*e64*sf64-factor_target['f'].float().double()
    expectedC=4*c64*e64[...,None,None]*rf64[...,None,None]*cf64/(N*stats['f_variance'])
    assert torch.allclose(gC.double(),expectedC,atol=2e-6,rtol=1e-5)
    assert torch.equal(gC[0,2],torch.zeros_like(gC[0,2])) and gFactorA[0,2].norm()>0
    assert gC.norm()>0 and torch.isfinite(gC).all()
    checks['analytic_C_gradient_max_abs']=float((gC.double()-expectedC).abs().max())
    checks['zero_C_nonzero_target_has_zero_factor_but_live_A_gradient']=True
    traceE=torch.autograd.grad(trace['trace'],energy,allow_unused=True,retain_graph=True)[0]
    assert traceE is None;checks['trace_intensity_has_no_direct_E_gradient']=True
    # Printed zero targets remain in the exact six-label mean.
    zeros=target['f']==0
    p32=energy*matrix.diagonal(dim1=-2,dim2=-1).sum(-1)*F_CONST
    assert torch.equal(raw['raw_f'],((p32-target['f'].float())**2).mean()/F_VARIANCE)
    assert float((p32[zeros]**2).sum())>0;checks['zero_target_count_included']=int(zeros.sum())
    # Two invalid sentinel labels; linear traces before mask stay compatible
    # with original base_loss, and candidate nonlinear arithmetic is selected.
    valid=mask.clone();valid[0,1]=False;valid[1,2]=False
    invalid=copy.deepcopy(target)
    for k in ('mask_E','mask_A','mask_f'):invalid[k]=valid
    e=energy.detach().clone();a=matrix.detach().clone();e[~valid]=float('nan');a[~valid]=float('nan')
    invalid['E'][~valid]=float('nan');invalid['A'][~valid]=float('nan');invalid['f'][~valid]=float('nan')
    e.requires_grad_();a.requires_grad_()
    for objective in ('trace','raw_f'):
        actual=loss_components((e,a),invalid,stats,objective)
        assert all(torch.isfinite(v) for v in actual.values())
        packed={k:v[valid] if k not in ('mask_E','mask_A','mask_f') else torch.ones(int(valid.sum()),dtype=torch.bool) for k,v in target.items()}
        expected=loss_components((energy[valid],matrix[valid]),packed,stats,objective)
        assert all(torch.equal(actual[k],expected[k]) for k in actual)
    invalidE,invalidA=torch.autograd.grad(actual['total'],(e,a))
    assert torch.isfinite(invalidE).all() and torch.isfinite(invalidA).all()
    assert torch.equal(invalidE[~valid],torch.zeros_like(invalidE[~valid]))
    assert torch.equal(invalidA[~valid],torch.zeros_like(invalidA[~valid]))
    checks['invalid_sentinels_have_finite_valid_and_zero_invalid_gradients']=True
    checks['invalid_sentinels_selected_before_nonlinear_arithmetic']=True
    empty=copy.deepcopy(target)
    empty['mask_f']=torch.zeros_like(mask)
    checks['empty_f_reject']=rejected(lambda:loss_components((energy,matrix),empty,stats,'raw_f'))
    empty['mask_E']=torch.zeros_like(mask)
    checks['empty_energy_reject']=rejected(lambda:loss_components((energy,matrix),empty,stats,'trace'))
    bad=copy.deepcopy(stats);bad['f_variance']=F_VARIANCE*2
    checks['changed_normalization_reject']=rejected(lambda:loss_components((energy,matrix),target,bad,'raw_f'))
    checks['wrong_objective_reject']=rejected(lambda:loss_components((energy,matrix),target,stats,'unknown'))
    checks['wrong_training_dtype_reject']=rejected(lambda:loss_components((energy.double(),matrix.double()),target,stats,'raw_f'))
    q=torch.tensor([[0.,1,0],[1,0,0],[0,0,1]],dtype=torch.float32) # reflection
    reflected=q@matrix@q.T;rot_target=copy.deepcopy(target);rot_target['A']=q@target['A']@q.T
    for objective in ('trace','raw_f'):
        value=loss_components((energy,reflected),rot_target,stats,objective)['total']
        assert torch.allclose(value,loss_components((energy,matrix),target,stats,objective)['total'],atol=2e-6,rtol=1e-5)
    checks['improper_O3_objective_invariance']=True
    torch.manual_seed(61);q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
    rotated=q@matrix@q.T;rot_target['A']=q@target['A']@q.T
    for objective in ('trace','raw_f'):
        assert torch.allclose(loss_components((energy,rotated),rot_target,stats,objective)['total'],
            loss_components((energy,matrix),target,stats,objective)['total'],atol=2e-6,rtol=1e-5)
    checks['proper_O3_objective_invariance']=True
    huge=energy.detach().clone();huge[0,0]=float('inf')
    assert not torch.isfinite(loss_components((huge,matrix),target,stats,'raw_f')['total'])
    checks['nonfinite_not_silently_repaired']=True
    hashes=[]
    for arm in ('trace_control','raw_f'):
        model=build_fresh(mc,stats,False)
        assert not model.adapter_enabled and not any(p.requires_grad for p in model.right_adapter.parameters())
        pair=(tensor_hash(base_state(model)),tensor_hash(model.state_dict()));hashes.append(pair)
        assert pair==(prior['initial_base_tensor_sha256'],prior['initial_full_tensor_sha256'])
        del model
    assert hashes[0]==hashes[1]
    names=['objective.py','synthetic_checks.py','fresh_model.py','common.py','config.json','model_config.json',
           'TRAIN_STATISTICS.json','TECHNICAL_PREFLIGHT_PLAN.md','ROUND06_PREPARATION_DECISION.md']
    receipt={'passed':True,'checks':checks,'synthetic_only':True,'model_updates':0,'raw_or_prediction_array_access':False,
        'trained_checkpoint_access':False,'fresh_model_constructions':2,
        'initial_base_tensor_sha256':hashes[0][0],'initial_full_tensor_sha256':hashes[0][1],
        'train_statistics_sha256':sha(ROOT/'TRAIN_STATISTICS.json'),
        'gradient_and_loss_tolerance':{'atol':2e-6,'rtol':1e-5},
        'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in names},
        'inherited_original_model_preflight_reference':str(CAMPAIGN/'round05_scratch_preparation/FROZEN_MANIFEST.json'),
        'inherited_original_model_preflight_manifest_sha256':sha(CAMPAIGN/'round05_scratch_preparation/FROZEN_MANIFEST.json'),
        'inherited82_current_pins_verified':True}
    immutable_json(receipt,ROOT/'CPU_PREFLIGHT.json');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
