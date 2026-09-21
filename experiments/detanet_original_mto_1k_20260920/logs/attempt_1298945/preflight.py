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
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare_tree(a[k],b[k],path+'/'+str(k),atol,rtol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for k,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,path+'/'+str(k),atol,rtol)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    else:assert a==b,path


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
        opt,scheduler=optimizer_scheduler(model,cfg)
        g=torch.Generator().manual_seed(11)
        update(model,opt,data['train'][:2],info['train_rms'],2,device)
        scheduler.step(1.)
        saved=copy.deepcopy(dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict(),rng=get_rng(g)))
        loss1=update(model,opt,data['train'][2:4],info['train_rms'],2,device)
        scheduler.step(.9)
        final=copy.deepcopy(dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict()))
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['opt'])
        scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],g)
        loss2=update(model,opt,data['train'][2:4],info['train_rms'],2,device);scheduler.step(.9)
        compare_tree(final,dict(model=model.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict()),atol=1e-6,rtol=1e-5)
        assert abs(loss1-loss2)<=1e-6+1e-5*abs(loss1)
        measurements[label]['optimizer_scheduler_rng_resume']=True
    del models
    torch.cuda.empty_cache()
    # Exercise the real trainer's mid-epoch interruption, including data cursor/order.
    root=ROOT/'reports/resume_audit'
    assert not root.exists(),'Use a fresh preflight attempt namespace'
    for variant in VARIANTS:
        train_one(11,variant,device,max_updates=3,output_root=root/'continuous',require_gate=False)
        train_one(11,variant,device,max_updates=1,output_root=root/'resumed',require_gate=False)
        train_one(11,variant,device,max_updates=2,output_root=root/'resumed',require_gate=False)
        left=torch.load(root/f'continuous/seed_11/{variant}/last.pt',map_location='cpu',weights_only=False)
        right=torch.load(root/f'resumed/seed_11/{variant}/last.pt',map_location='cpu',weights_only=False)
        for key in ['model','optimizer','scheduler','rng']:compare_tree(left[key],right[key],key,atol=1e-6,rtol=1e-5)
        for key in ['step','epoch','order','cursor']:assert left['progress'][key]==right['progress'][key]
        measurements[variant]['trainer_mid_epoch_resume']=True
    atomic_json(measurements,ROOT/'reports/resume_cpu_gpu.json')


def smoke(cfg,data,info,device):
    records=data['train'][:32]
    assert [r['id'] for r in records]==info['smoke_ids']
    models=build_triplet(cfg,info['n_ref'],info['grid'],11)
    results={}
    for variant,model in models.items():
        model.to(device);opt,_=optimizer_scheduler(model,cfg)
        started=time.monotonic();history=[]
        initial,_=evaluate(model,records,info['train_rms'],cfg['micro_batch_size'],device)
        best=initial
        for step in range(cfg['smoke_max_steps']):
            update(model,opt,records,info['train_rms'],cfg['micro_batch_size'],device)
            if (step+1)%10==0:
                loss,_=evaluate(model,records,info['train_rms'],cfg['micro_batch_size'],device)
                best=min(best,loss);history.append(dict(step=step+1,mse=loss,best=best))
                print('SMOKE',variant,step+1,loss,best,flush=True)
                if best<cfg['smoke_best_mse_below'] and best<cfg['smoke_best_initial_ratio_below']*initial:break
        passed=best<cfg['smoke_best_mse_below'] and best<cfg['smoke_best_initial_ratio_below']*initial
        result=dict(initial=initial,best=best,ratio=best/initial,steps=step+1,passed=passed,
                    seconds=time.monotonic()-started,history=history)
        if variant!=VARIANTS[0]:
            x,_=collate(records[:2],device)
            with torch.no_grad():_,e=model(**x,export=True)
            result['diagnostics']=dict(gate_saturation=float((e['c'].abs()>.99).float().mean()),
                energy_range=[float(e['E'].min()),float(e['E'].max())],
                effective_states=float((e['f']>e['f'].max(-1,keepdim=True).values*.01).float().sum(-1).mean()),
                tensor_rms=float(e['Q'].square().mean().sqrt()),
                beta_rms=float(e['beta'].square().mean().sqrt()))
        results[variant]=result;atomic_json(results,ROOT/'reports/smoke_seed11.json')
        model.cpu();torch.cuda.empty_cache()
        # Finish independent arms even if one smoke gate fails, then fail the collective gate.
    assert all(v['passed'] for v in results.values()),'32-molecule smoke gate failed; formal training forbidden'


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU allocation required'
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/'protocol.json').read_text());data,info=load()
    for dtype in ['float64','float32']:
        report=json.loads((ROOT/f'logs/acceptance_cuda_{dtype}.json').read_text())
        assert report['success']
    cpu_gpu_and_resume(cfg,data,info,'cuda')
    oracle(data,info,'cuda')
    smoke(cfg,data,info,'cuda')
    atomic_json(dict(fingerprint=fingerprint(),job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
                     seed11_end_to_end=True,smoke_checkpoint_used_for_training=False),ROOT/'reports/PREFLIGHT_PASSED.json')


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(dict(traceback=traceback.format_exc(),job_id=os.getenv('SLURM_JOB_ID'),
                         formal_training_allowed=False),ROOT/'reports/PREFLIGHT_FAILED.json')
        raise
