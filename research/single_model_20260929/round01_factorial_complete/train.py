"""One checkpoint per arm, validation-only matched 2x2 continuation pilot."""
import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from architecture.model import SOURCE,build_model,load_baseline,training_loss
from metrics import validate
from freeze import sha,current_hashes
sys.path.insert(0,str(SOURCE))
from dataset import Data

ROOT=Path(__file__).resolve().parent
STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def atomic_json(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def atomic_torch(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_arrays(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**value)
    os.replace(tmp,path)

def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'order':generator.bit_generator.state,'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state()}

def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    generator.bit_generator.state=state['order']
    torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state(state['cuda'].cpu())

def setup(cfg):
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(cfg['num_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def run(arm):
    cfg=json.loads((ROOT/'round_config.json').read_text())
    assert not cfg['allow_test_evaluation'] and cfg['dtype']=='float32' and not cfg['amp']
    ac=cfg['arms'][arm]
    out=ROOT/'runs'/arm
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    assert frozen['config']==cfg
    hashes=current_hashes(cfg)
    assert hashes==frozen['source_hashes'],'Source/data changed after freeze'
    manifest_sha=sha(ROOT/'FROZEN_MANIFEST.json')
    if (out/'FIT_COMPLETE.json').exists():
        assert json.loads((out/'FIT_COMPLETE.json').read_text())['manifest_sha256']==manifest_sha
        print('Already complete',arm,flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    setup(cfg)
    data=Data(device='cuda')
    assert len(data.parts['train'])==frozen['train_count']
    assert len(data.parts['val'])==frozen['validation_count']
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
        assert np.array_equal(z['ids'],data.ids)
    source_cfg=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    original=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    assert original['epoch']==33 and original['config']==source_cfg
    model=load_baseline(build_model(source_cfg,data.stats,ac['adapter'],cfg['adapter_seed']),original['model']).cuda()
    del original
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'],amsgrad=True,weight_decay=cfg['weight_decay'])
    # Explicit reset after construction makes arm-specific initialization unable
    # to change runtime RNG. The order generator is independent of Torch RNG.
    setup(cfg)
    generator=np.random.default_rng(cfg['order_seed'])
    state={'completed_epoch':0,'history':[],'best_epoch':0,'best_sse':None,'best_metrics':None,
           'epoch_zero_metrics':None,'steps':0}
    def save_last():
        atomic_torch({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':rng_state(generator),'state':state,'config':cfg,'arm':arm,
            'manifest_sha256':manifest_sha},out/'last.pt')
    def save_best(epoch,result,arrays):
        atomic_torch({'model':model.state_dict(),'epoch':epoch,'validation':result,
            'source_config':source_cfg,'arm_config':ac,'adapter_seed':cfg['adapter_seed'],
            'manifest_sha256':manifest_sha,'arm':arm},out/'best.pt')
        save_arrays(arrays,out/'best_validation_predictions.npz')
        atomic_json({'epoch':epoch,'validation':result,'manifest_sha256':manifest_sha},out/'best_metrics.json')
    if (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        assert last['config']==cfg and last['arm']==arm and last['manifest_sha256']==manifest_sha
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        state=last['state'];restore_rng(last['rng'],generator)
        del last
        # Best checkpoint may have been updated after the most recent durable
        # epoch save. Restore it from the epoch state when this happened.
        durable=out/('selected_epoch_%03d.pt'%state['best_epoch'])
        assert durable.exists()
        selected=torch.load(durable,map_location='cpu',weights_only=False)
        atomic_torch(selected,out/'best.pt')
        atomic_json({'epoch':state['best_epoch'],'validation':state['best_metrics'],
                     'manifest_sha256':manifest_sha},out/'best_metrics.json')
        del selected
        with (out/'history.jsonl').open('w') as f:
            for row in state['history']:
                f.write(json.dumps(row,allow_nan=False)+'\n')
    else:
        # Orphan startup artifacts are recoverable because no fit occurred yet.
        result,arrays=validate(model,data,raw,cfg['batch_size'],frozen['tail_thresholds'])
        anchor=result['raw_f']['r2']
        assert abs(anchor-cfg['epoch_zero_anchor_r2'])<cfg['epoch_zero_anchor_r2_atol'],('Epoch-zero replay failed',anchor)
        state.update(best_sse=result['raw_f']['sse'],best_metrics=result,epoch_zero_metrics=result)
        row={'epoch':0,'validation':result,'elapsed_seconds':0.0}
        state['history'].append(row)
        save_best(0,result,arrays)
        atomic_torch(torch.load(out/'best.pt',map_location='cpu',weights_only=False),out/'selected_epoch_000.pt')
        save_last()
        (out/'history.jsonl').write_text(json.dumps(row,allow_nan=False)+'\n')
    atomic_json({'arm':arm,'config':cfg,'arm_config':ac,'manifest_sha256':manifest_sha,
        'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name':torch.cuda.get_device_name(0),'started_at_unix':time.time(),
        'resume_completed_epoch':state['completed_epoch'],
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)},out/'run_manifest.json')
    print(json.dumps({'arm':arm,'event':'ready','epoch0_r2':state['epoch_zero_metrics']['raw_f']['r2']}),flush=True)
    last_status=0
    for epoch in range(state['completed_epoch']+1,cfg['epochs']+1):
        if STOP:
            break
        started=time.monotonic()
        order=generator.permutation(data.parts['train'])
        order_sha=hashlib.sha256(order.tobytes()).hexdigest()
        model.train()
        totals=np.zeros(4,dtype=np.float64);count=0
        max_grad_norm=0.0
        for cursor in range(0,len(order),cfg['batch_size']):
            indices=order[cursor:cursor+cfg['batch_size']]
            x,y=data.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            pred,values=training_loss(model,x,y,data.stats,ac['orth_lambda'])
            loss=values['total']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite loss at epoch %d cursor %d'%(epoch,cursor))
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            max_grad_norm=max(max_grad_norm,float(grad_norm))
            totals+=np.array([float(values[k].detach()) for k in ('total','energy','trace','decorrelation')])*len(indices)
            count+=len(indices);state['steps']+=1
            if time.monotonic()-last_status>60:
                atomic_json({'state':'running','pid':os.getpid(),'arm':arm,'epoch':epoch,
                    'completed_epoch':state['completed_epoch'],'cursor':cursor+len(indices),
                    'train_count':len(order),'last_update_unix':time.time(),
                    'best_epoch':state['best_epoch'],'best_r2':state['best_metrics']['raw_f']['r2'],
                    'manifest_sha256':manifest_sha},out/'status.json')
                last_status=time.monotonic()
        result,arrays=validate(model,data,raw,cfg['batch_size'],frozen['tail_thresholds'])
        if result['raw_f']['sse']<state['best_sse']:
            state.update(best_sse=result['raw_f']['sse'],best_epoch=epoch,best_metrics=result)
            save_best(epoch,result,arrays)
            # Retain each selected epoch until closeout so an interrupted
            # transaction can restore the best matching durable epoch state.
            atomic_torch(torch.load(out/'best.pt',map_location='cpu',weights_only=False),
                         out/('selected_epoch_%03d.pt'%epoch))
        state['completed_epoch']=epoch
        row={'epoch':epoch,'validation':result,'train':dict(zip(('total','energy','trace','decorrelation'),(totals/count).tolist())),
            'order_sha256':order_sha,'seconds':time.monotonic()-started,'max_preclip_gradient_norm':max_grad_norm,
            'best_epoch':state['best_epoch'],'best_r2':state['best_metrics']['raw_f']['r2']}
        state['history'].append(row)
        save_last()
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        atomic_json({'state':'epoch_complete','pid':os.getpid(),'arm':arm,'epoch':epoch,
            'completed_epoch':epoch,'last_update_unix':time.time(),'best_epoch':state['best_epoch'],
            'best_r2':state['best_metrics']['raw_f']['r2'],'manifest_sha256':manifest_sha},out/'status.json')
        print(json.dumps({'arm':arm,'epoch':epoch,'raw_f_r2':result['raw_f']['r2'],
            'best_epoch':state['best_epoch'],'best_r2':state['best_metrics']['raw_f']['r2'],
            'seconds':row['seconds'],'order_sha256':order_sha}),flush=True)
    if state['completed_epoch']==cfg['epochs']:
        # Re-evaluate exactly the single selected checkpoint. This also repairs
        # prediction arrays after a resume transaction, without selecting anew.
        selected=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(selected['model'])
        final,arrays=validate(model,data,raw,cfg['batch_size'],frozen['tail_thresholds'])
        assert abs(final['raw_f']['r2']-state['best_metrics']['raw_f']['r2'])<1e-7
        save_arrays(arrays,out/'best_validation_predictions.npz')
        final_result={'arm':arm,'manifest_sha256':manifest_sha,'completed_epoch':state['completed_epoch'],
            'selected_epoch':state['best_epoch'],'validation':final,'epoch_zero':state['epoch_zero_metrics'],
            'best_checkpoint':str(out/'best.pt'),'best_checkpoint_sha256':sha(out/'best.pt'),
            'finished_at_unix':time.time(),'single_checkpoint':True,'test_evaluated':False}
        atomic_json(final_result,out/'FIT_COMPLETE.json')
        atomic_json({'state':'complete','arm':arm,'last_update_unix':time.time(),**final_result},out/'status.json')
    else:
        atomic_json({'state':'stopped_resumable','arm':arm,'completed_epoch':state['completed_epoch'],
            'last_update_unix':time.time(),'manifest_sha256':manifest_sha},out/'status.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,
        choices=['control','adapter','decorrelation','both']);args=parser.parse_args()
    try:
        run(args.arm)
    except Exception as exc:
        out=ROOT/'runs'/args.arm;out.mkdir(parents=True,exist_ok=True)
        atomic_json({'arm':args.arm,'exception':repr(exc),'traceback':traceback.format_exc(),
                     'failed_at_unix':time.time()},out/'FAILED.json')
        raise

if __name__=='__main__':
    main()
