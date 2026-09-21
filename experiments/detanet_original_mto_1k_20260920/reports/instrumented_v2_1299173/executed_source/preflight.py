"""Allocated-GPU acceptance, decoder oracle, strict smoke, and resume audit."""
import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
import numpy as np
import torch
from data_protocol import ROOT,load,collate
from models import build_triplet,VARIANTS,PhysicalSpectrumDecoder
from train import (fingerprint,atomic_json,atomic_save,optimizer_scheduler,update,evaluate,
                   get_rng,restore_rng,train_one)


def fit_free_peaks(target,grid,starts=3,steps=800):
    """Ten free E/f; local optimization diagnostic, never a certified lower bound."""
    results=[]
    for seed in range(starts):
        gen=torch.Generator(device=target.device).manual_seed(9000+seed)
        initial=torch.linspace(2.,13.,10,device=target.device)[None].expand(len(target),-1).clone()
        initial+=(torch.rand(initial.shape,device=target.device,generator=gen)-.5)*.8
        raw_e=torch.nn.Parameter(torch.log(torch.expm1(initial)))
        raw_f=torch.nn.Parameter(torch.full_like(initial,-3.))
        opt=torch.optim.Adam([raw_e,raw_f],lr=.03)
        best=float('inf');history=[];best_values=None
        scale=target.square().mean().sqrt()
        for step in range(steps):
            opt.zero_grad();E=torch.nn.functional.softplus(raw_e);f=torch.nn.functional.softplus(raw_f)
            pred=(f[...,None]*torch.exp(-.5*((grid-E[...,None])/.2).square())/(.2*np.sqrt(2*np.pi))).sum(-2)
            loss=((pred-target)/scale).square().mean()
            value=float(loss.detach())
            if value<best:best=value;best_values=dict(E=E.detach().cpu(),f=f.detach().cpu(),prediction=pred.detach().cpu())
            loss.backward();opt.step()
            if step%100==0:history.append([step,value])
        results.append(dict(seed=seed,best_normalized_mse=best,history=history,**best_values))
    return results


def oracle(data,info,device):
    records=data['train'][:4]
    assert [r['id'] for r in records]==info['oracle_ids']
    targets=torch.stack([r['spectrum'] for r in records]).to(device)
    grid=torch.tensor(info['grid'],device=device,dtype=torch.float32)
    real=fit_free_peaks(targets,grid)
    E=torch.linspace(2.,13.,10,device=device)[None]
    f=torch.linspace(.01,.08,10,device=device)[None]
    synthetic=(f[...,None]*torch.exp(-.5*((grid-E[...,None])/.2).square())/(.2*np.sqrt(2*np.pi))).sum(-2)
    synth=fit_free_peaks(synthetic,grid,starts=2,steps=1000)
    assert min(r['best_normalized_mse'] for r in synth)<1e-4
    atomic_save(dict(ids=info['oracle_ids'],real=real,synthetic=synth),ROOT/'reports/decoder_oracle.pt')
    report=dict(ids=info['oracle_ids'],sigma_eV=.2,states=10,
        real=[{k:v for k,v in r.items() if not torch.is_tensor(v)} for r in real],
        synthetic=[{k:v for k,v in r.items() if not torch.is_tensor(v)} for r in synth],
        interpretation='training samples only; best local fit is NOT a global error lower bound; no policy selected')
    atomic_json(report,ROOT/'reports/decoder_oracle.json')
    print('ORACLE',json.dumps(report),flush=True)


def compare_tree(a,b,path='root',atol=0.,rtol=0.):
    if torch.is_tensor(a):
        try:torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        except AssertionError as exc:raise AssertionError(path+'\n'+str(exc)) from exc
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare_tree(a[k],b[k],path+'/'+str(k),atol,rtol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for k,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,path+'/'+str(k),atol,rtol)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    else:assert a==b,path


def adam_moment_contract(before,after,gradients):
    """Verify each replay against its actual gradient, including FP32 rounding.

    CUDA atomic scatter is nondeterministic, so different subsequent gradients
    need not produce bitwise equal optimizer moments. Loaded state itself must.
    """
    beta1,beta2=before['param_groups'][0]['betas']
    errors={}
    for pid,state in after['state'].items():
        if pid not in gradients:continue
        old=before['state'][pid];g=gradients[pid]
        expected_m=beta1*old['exp_avg']+(1-beta1)*g
        expected_v=beta2*old['exp_avg_sq']+(1-beta2)*g.square()
        expected_max=torch.maximum(old['max_exp_avg_sq'],expected_v)
        for key,want in [('exp_avg',expected_m),('exp_avg_sq',expected_v),('max_exp_avg_sq',expected_max)]:
            actual=state[key];err=(actual-want).abs()
            # A small arithmetic-rounding bound, not a fitted tolerance.
            magnitude=(beta1*old['exp_avg']).abs()+((1-beta1)*g).abs() if key=='exp_avg' else old['max_exp_avg_sq'].abs()+expected_v.abs()
            bound=8*torch.finfo(actual.dtype).eps*magnitude.clamp_min(torch.finfo(actual.dtype).tiny)
            assert (err<=bound).all(),(pid,key,float(err.max()),float(bound.max()))
            errors[f'{pid}/{key}']=float(err.max())
    return max(errors.values(),default=0.)


def cpu_gpu_and_resume(cfg,data,info,device):
    models=build_triplet(cfg,info['n_ref'],info['grid'],11)
    measurements={}
    for label,model in models.items():
        x,_=collate(data['train'][:2],'cpu')
        with torch.no_grad():cpu=model(**x)
        model.to(device);x,_=collate(data['train'][:2],device)
        with torch.no_grad():gpu=model(**x).cpu()
        torch.testing.assert_close(cpu,gpu,atol=1e-5,rtol=1e-4)
        measurements[label]=dict(cpu_gpu_max_abs=float((cpu-gpu).abs().max()))
        # Exercise the actual common effective batch before any formal submission.
        torch.cuda.reset_peak_memory_stats()
        full_x,full_y=collate(data['train'][:cfg['effective_batch_size']],device)
        full_loss=((model(**full_x)-full_y)/info['train_rms']).square().mean()
        full_loss.backward()
        assert torch.isfinite(full_loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        measurements[label]['batch64_forward_backward_peak_bytes']=torch.cuda.max_memory_allocated()
        model.zero_grad(set_to_none=True)
        del full_x,full_y,full_loss
        opt,scheduler=optimizer_scheduler(model,cfg)
        g=torch.Generator().manual_seed(11)
        update(model,opt,data['train'][:2],info['train_rms'],2,device)
        scheduler.step(1.)
        saved=copy.deepcopy(dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict(),rng=get_rng(g)))
        atomic_save(saved,ROOT/f'reports/gpu_resume_{label}.pt')
        loss1=update(model,opt,data['train'][2:4],info['train_rms'],2,device)
        grad1={i:p.grad.detach().clone() for i,p in enumerate(model.parameters()) if p.grad is not None}
        scheduler.step(.9)
        final=copy.deepcopy(dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict()))
        saved=torch.load(ROOT/f'reports/gpu_resume_{label}.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['opt'])
        scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],g)
        compare_tree(saved['model'],model.state_dict())
        compare_tree(saved['opt'],opt.state_dict())
        compare_tree(saved['scheduler'],scheduler.state_dict())
        # Snapshot load_state_dict tensors before the optimizer mutates its state.
        before=copy.deepcopy(opt.state_dict())
        loss2=update(model,opt,data['train'][2:4],info['train_rms'],2,device);scheduler.step(.9)
        grad2={i:p.grad.detach().clone() for i,p in enumerate(model.parameters()) if p.grad is not None}
        compare_tree(final['model'],model.state_dict(),atol=1e-6,rtol=1e-5)
        compare_tree(final['scheduler'],scheduler.state_dict())
        measurements[label]['adam_replay_roundoff_error']=[adam_moment_contract(before,final['opt'],grad1),adam_moment_contract(before,opt.state_dict(),grad2)]
        measurements[label]['next_backward_gradient_max_abs_difference']=max(float((grad1[k]-grad2[k]).abs().max()) for k in grad1)
        measurements[label]['loaded_optimizer_model_scheduler_bitwise_equal']=True
        assert abs(loss1-loss2)<=1e-6+1e-5*abs(loss1)
        measurements[label]['optimizer_scheduler_rng_resume']=True
    del models
    torch.cuda.empty_cache()
    # Exercise the real trainer's mid-epoch interruption, including data cursor/order.
    root=ROOT/f'reports/resume_audit_{os.environ["SLURM_JOB_ID"]}'
    assert not root.exists(),'Use a fresh preflight attempt namespace'
    # The complete multi-step trainer/data-cursor replay is checked on deterministic
    # CPU scatter in the allocated job. GPU disk restore and next step were checked above.
    replay_device='cpu'
    for variant in VARIANTS:
        train_one(11,variant,replay_device,max_updates=3,output_root=root/'continuous',require_gate=False)
        train_one(11,variant,replay_device,max_updates=1,output_root=root/'resumed',require_gate=False)
        train_one(11,variant,replay_device,max_updates=2,output_root=root/'resumed',require_gate=False)
        left=torch.load(root/f'continuous/seed_11/{variant}/last.pt',map_location='cpu',weights_only=False)
        right=torch.load(root/f'resumed/seed_11/{variant}/last.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','scheduler','rng']:compare_tree(left[key],right[key],key)
        for key in ['step','epoch','order','cursor']:assert left['progress'][key]==right['progress'][key]
        measurements[variant]['trainer_mid_epoch_resume']=True
        measurements[variant]['trainer_replay_device']='cpu'
    atomic_json(measurements,ROOT/'reports/resume_cpu_gpu.json')


def smoke(cfg,data,info,device):
    # Main optimization is unchanged; diagnostics are isolated and never satisfy this gate.
    from smoke_diagnostics import run
    return run(cfg,data,info,device)


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU allocation required'
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/'protocol.json').read_text());data,info=load()
    for dtype in ['float64','float32']:
        report=json.loads((ROOT/f'logs/acceptance_cuda_{dtype}.json').read_text())
        assert report['success']
    failures={}
    for name,fn in [('oracle',lambda:oracle(data,info,'cuda')),
                    ('resume',lambda:cpu_gpu_and_resume(cfg,data,info,'cuda')),
                    ('smoke',lambda:smoke(cfg,data,info,'cuda'))]:
        try:fn()
        except Exception:
            failures[name]=traceback.format_exc()
            print('CHECK_FAILED',name,failures[name],flush=True)
            atomic_json(failures,ROOT/'reports/preflight_subcheck_failures.json')
    assert not failures,'Formal training blocked by '+','.join(failures)
    atomic_json(dict(fingerprint=fingerprint(),job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
                     seed11_end_to_end=True,smoke_checkpoint_used_for_training=False),ROOT/'reports/PREFLIGHT_PASSED.json')


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(dict(traceback=traceback.format_exc(),job_id=os.getenv('SLURM_JOB_ID'),
                         formal_training_allowed=False),ROOT/'reports/PREFLIGHT_FAILED.json')
        raise
