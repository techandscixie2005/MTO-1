"""Single seed A/B training for each existing scale; resume only this run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
from data_protocol import ROOT, load, collate, sha
from models import build_pair, VARIANTS, UV

STOP = False


def request_stop(*args):
    global STOP
    STOP = True


def fingerprint(scale):
    paths = sorted(ROOT.glob('*.py')) + sorted((ROOT/'tests').glob('*.py')) + sorted((ROOT/'jobs').glob('*.slurm'))
    paths += [ROOT/'protocol.json', ROOT/f'data/frozen_{scale}.json'] + sorted((ROOT/'vendor').rglob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        torch.save(obj, f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_json(obj,path):
    path=Path(path);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)


def get_rng(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                shuffle=generator.get_state())


def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    generator.set_state(state['shuffle'].cpu())


def optimizer_scheduler(model,cfg):
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],amsgrad=cfg['amsgrad'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=cfg['factor'],patience=cfg['patience'],
        min_lr=cfg['min_lr'],threshold=cfg['scheduler_threshold'],threshold_mode=cfg['scheduler_threshold_mode'])
    return opt,scheduler


def update(model,opt,records,scale,micro,device):
    """Every molecule has equal weight, including last microbatch/last effective batch."""
    opt.zero_grad(set_to_none=True)
    total=0.
    for start in range(0,len(records),micro):
        sample=records[start:start+micro]
        x,y=collate(sample,device)
        loss=((model(**x)-y)/scale).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite spectrum loss')
        weighted=loss*(len(sample)/len(records));weighted.backward()
        total+=float(weighted.detach())
    for name,p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient '+name)
    return total


def record_oom(state,out,device):
    old=state['micro_batch_size']
    if old<=1:raise RuntimeError('OOM even at micro batch 1')
    state['micro_batch_size']=max(1,old//2)
    state['oom_adjustments'].append(dict(step=state.get('step',0),old=old,new=state['micro_batch_size'],effective_batch=64))
    atomic_json(state['oom_adjustments'],out/'oom_adjustments.json')
    import gc
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    print(json.dumps(dict(event='OOM_REDUCE_MICRO',**state['oom_adjustments'][-1])),flush=True)

def adaptive_update(model,opt,records,scale,state,device,out):
    while True:
        try:
            loss=update(model,opt,records,scale,state['micro_batch_size'],device)
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            record_oom(state,out,device)
    opt.step()
    return loss

def adaptive_evaluate(model,records,scale,state,device,out):
    while True:
        try:return evaluate(model,records,scale,state['micro_batch_size'],device)
        except torch.cuda.OutOfMemoryError:record_oom(state,out,device)

def short_check(scale,seed,variant,data,info,cfg,device,out,fp):
    models=build_pair(cfg,info['n_ref'],info['grid'],seed)
    model=models[variant].to(device);del models
    x,y=collate(data['train'][:2],device)
    prediction=model(**x)
    assert prediction.shape==y.shape and torch.isfinite(prediction).all()
    loss=((prediction-y)/info['train_rms']).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    atomic_json(dict(scale=scale,seed=seed,variant=variant,passed=True,finite_loss=float(loss.detach()),
        gradients_finite=True,molecules=2,optimizer_updates=0,weights_discarded=True,
        performance_threshold=None,fingerprint=fp),out/'short_check.json')
    del model,x,y,prediction,loss,grads
    import gc
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model,records,scale,micro,device):
    model.eval();total=0.;outputs=[]
    for start in range(0,len(records),micro):
        sample=records[start:start+micro];x,y=collate(sample,device)
        pred=model(**x)
        total+=float(((pred-y)/scale).square().mean())*len(sample)
        outputs.append(pred.cpu())
    model.train()
    return total/len(records),torch.cat(outputs)


def checkpoint(model,opt,scheduler,generator,state,fp):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                rng=get_rng(generator),progress=state,fingerprint=fp,uv_constructor=UV)


def restore(path,model,opt,scheduler,generator,fp,device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    if ckpt['fingerprint']!=fp:raise RuntimeError('Refuse resume: code/config/data fingerprint mismatch')
    model.load_state_dict(ckpt['model'],strict=True)
    opt.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
    restore_rng(ckpt['rng'],generator)
    return ckpt['progress']


def train_one(scale,seed,variant,device='cuda',max_updates=None,output_root=None):
    cfg=json.loads((ROOT/'protocol.json').read_text());data,info=load(scale);fp=fingerprint(scale)
    from workflow import require_previous
    require_previous(scale)
    assert cfg['seeds']==[11] and tuple(cfg['variants'])==VARIANTS
    out=(ROOT/'runs'/scale if output_root is None else Path(output_root))/f'seed_{seed}'/variant
    out.mkdir(parents=True,exist_ok=True)
    if (out/'FIT_COMPLETE.json').exists():
        done=json.loads((out/'FIT_COMPLETE.json').read_text());assert done['fingerprint']==fp
        return done
    if not (out/'last.pt').exists():
        short_check(scale,seed,variant,data,info,cfg,device,out,fp)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    models=build_pair(cfg,info['n_ref'],info['grid'],seed)
    model=models[variant].to(device);del models
    opt,scheduler=optimizer_scheduler(model,cfg)
    # Independent of construction RNG and identical for each arm of a seed.
    generator=torch.Generator().manual_seed(seed+1000003)
    state=dict(scale=scale,seed=seed,variant=variant,epoch=0,step=0,order=None,cursor=0,best=float('inf'),
               best_epoch=-1,bad_epochs=0,lr_reductions=0,last_reduction=-1,history=[],
               epoch_loss_sum=0.,epoch_count=0,elapsed_seconds=0.,micro_batch_size=cfg['micro_batch_size'],oom_adjustments=[])
    if (out/'last.pt').exists():
        state=restore(out/'last.pt',model,opt,scheduler,generator,fp,device)
        if state['scale'] != scale or state['seed'] != seed or state['variant'] != variant:
            raise RuntimeError('Refuse resume: seed/variant mismatch')
        atomic_json(state['history'],out/'history.json')
    started=time.monotonic();steps_this_call=0
    batch=cfg['effective_batch_size'];micro=state['micro_batch_size'];train=data['train']
    def save_last():
        state['elapsed_seconds']+=time.monotonic()-save_last.last
        save_last.last=time.monotonic()
        atomic_save(checkpoint(model,opt,scheduler,generator,state,fp),out/'last.pt')
    save_last.last=started
    for epoch in range(state['epoch'],cfg['max_epochs']):
        if state.get('training_complete'):break
        if state['order'] is None:state['order']=torch.randperm(len(train),generator=generator).tolist()
        for start in range(state['cursor'],len(train),batch):
            records=[train[i] for i in state['order'][start:start+batch]]
            loss=adaptive_update(model,opt,records,info['train_rms'],state,device,out)
            micro=state['micro_batch_size']
            state['step']+=1;steps_this_call+=1;state['cursor']=start+len(records)
            state['epoch_loss_sum']+=loss*len(records);state['epoch_count']+=len(records)
            if state['step'] % 50 == 0:save_last()
            if STOP or (max_updates is not None and steps_this_call>=max_updates):
                save_last();return dict(interrupted=True,step=state['step'],epoch=epoch)
        val,_=adaptive_evaluate(model,data['validation'],info['train_rms'],state,device,out)
        micro=state['micro_batch_size']
        if not np.isfinite(val):raise FloatingPointError('nonfinite validation loss')
        old_lr=opt.param_groups[0]['lr'];scheduler.step(val);lr=opt.param_groups[0]['lr']
        if lr<old_lr:state['lr_reductions']+=1;state['last_reduction']=epoch
        improved=val<state['best']
        if improved:state.update(best=val,best_epoch=epoch,bad_epochs=0)
        else:state['bad_epochs']+=1
        row=dict(epoch=epoch+1,step=state['step'],train_mse=state['epoch_loss_sum']/state['epoch_count'],
                 validation_mse=val,lr=lr,lr_reductions=state['lr_reductions'],
                 elapsed_seconds=state['elapsed_seconds']+time.monotonic()-save_last.last)
        state['history'].append(row)
        print(json.dumps(dict(seed=seed,variant=variant,**row)),flush=True)
        state.update(epoch=epoch+1,order=None,cursor=0,epoch_loss_sum=0.,epoch_count=0)
        can_stop=(state['epoch']>=cfg['min_epochs'] and state['bad_epochs']>=cfg['early_stopping_patience']
                  and state['lr_reductions']>=cfg['early_stop_min_actual_lr_reductions']
                  and epoch-state['last_reduction']>=cfg['early_stop_epochs_after_last_reduction'])
        state['training_complete']=can_stop or state['epoch']==cfg['max_epochs']
        if improved:atomic_save(checkpoint(model,opt,scheduler,generator,state,fp),out/'best.pt')
        save_last()
        atomic_json(state['history'],out/'history.json')
        if can_stop:break
        if STOP:return dict(interrupted=True,step=state['step'],epoch=state['epoch'])
    assert state.get('training_complete'), 'Refuse to mark interrupted training complete'
    done=dict(scale=scale,seed=seed,variant=variant,best_epoch=state['best_epoch']+1,best_validation_mse=state['best'],
              epochs=state['epoch'],optimizer_steps=state['step'],seconds=state['elapsed_seconds'],
              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
              max_epochs_reached=state['epoch']==cfg['max_epochs'],last_improvement_epochs_ago=state['bad_epochs'],
              micro_batch_size=state['micro_batch_size'],oom_adjustments=state['oom_adjustments'],fingerprint=fp,job_id=os.getenv('SLURM_JOB_ID'),node=os.uname().nodename)
    atomic_json(done,out/'FIT_COMPLETE.json')
    return done


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scale',choices=['1k','10k','full'],required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--variant',choices=VARIANTS,required=True);args=ap.parse_args()
    assert args.seed in json.loads((ROOT/'protocol.json').read_text())['seeds']
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    result=train_one(args.scale,args.seed,args.variant)
    print(json.dumps(result),flush=True)
    if result.get('interrupted'):sys.exit(75)


if __name__=='__main__':main()
