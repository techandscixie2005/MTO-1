"""Matched frozen-base F-only training; cache is never an inference dependency."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from model_frozen import SOURCE,build,frozen_digests,assert_optimizer,losses,movement
from cache_access import load_cache,targets
from train import atomic_json,atomic_torch,save_arrays,setup,rng_state,restore_rng
from metrics import validate
from freeze import sha
from dataset import Data
STOP=False

def stop(*_):
    global STOP
    STOP=True

def run(arm):
    cfg=json.loads((ROOT/'config.json').read_text());assert arm in cfg['arms']
    pc=json.loads((PARENT/'round_config.json').read_text())
    mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text());mh=sha(ROOT/'FROZEN_MANIFEST.json')
    assert mf['config']==cfg
    for path,value in mf['source_hashes'].items():assert sha(path)==value,('Changed frozen source',path)
    assert sha(ROOT/'CACHE_COMPLETE.json')==mf['cache_receipt_sha256']
    rd=ROOT/'runs'/arm;rd.mkdir(parents=True,exist_ok=True)
    lock=(rd/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (rd/'FIT_COMPLETE.json').exists():
        assert json.loads((rd/'FIT_COMPLETE.json').read_text())['manifest_sha256']==mh
        print('Already complete',arm,flush=True);return
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    setup(cfg);data=Data('cuda');cache,indices,cache_receipt=load_cache(data)
    sc=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    original=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)
    model=build(sc,data.stats,original['model'],cfg['adapter_seed']).cuda();del original
    expected_frozen=cache_receipt['frozen_parameter_buffer_hashes']
    assert frozen_digests(model)==expected_frozen
    opt=torch.optim.Adam(model.right_adapter.parameters(),lr=cfg['lr'],amsgrad=True,weight_decay=cfg['weight_decay'])
    assert_optimizer(model,opt)
    setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        assert np.array_equal(z['ids'],data.ids)
        raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
    thresholds=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())['tail_thresholds']
    chosen=np.random.default_rng(20260930).choice(indices,256,replace=False)
    inverse={int(v):i for i,v in enumerate(indices)}
    diagnostic_rows=torch.as_tensor([inverse[int(i)] for i in chosen],device='cuda')
    diagnostic_m={t:a[diagnostic_rows] for t,a in cache.items()}
    state={'completed_epoch':0,'history':[],'best_epoch':0,'best_sse':None,'best_metrics':None,
           'epoch_zero_metrics':None,'steps':0}
    def save_last():
        atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator),
            'state':state,'config':cfg,'source_config':sc,'stats':data.stats,'arm':arm,
            'manifest_sha256':mh,'cache_receipt_sha256':mf['cache_receipt_sha256'],
            'frozen_parameter_buffer_hashes':expected_frozen},rd/'last.pt')
    def save_best(epoch,metrics,arrays):
        ck={'model':model.state_dict(),'epoch':epoch,'validation':metrics,'config':cfg,
            'source_config':sc,'stats':data.stats,'arm':arm,'adapter_enabled':True,
            'manifest_sha256':mh,'geometry_only_inference':True,
            'frozen_parameter_buffer_hashes':expected_frozen}
        atomic_torch(ck,rd/('selected_epoch_%03d.pt'%epoch))
        temp=rd/'best.pt.tmp';shutil.copyfile(rd/('selected_epoch_%03d.pt'%epoch),temp);os.replace(temp,rd/'best.pt')
        save_arrays(arrays,rd/'best_validation_predictions.npz')
        atomic_json({'epoch':epoch,'validation':metrics},rd/'best_metrics.json')
    if (rd/'last.pt').exists():
        # Let Adam load move moment tensors to parameter devices while keeping
        # its non-capturable scalar step counters on CPU, as at fresh creation.
        ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
        assert ck['config']==cfg and ck['arm']==arm and ck['manifest_sha256']==mh
        assert ck['cache_receipt_sha256']==mf['cache_receipt_sha256']
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);state=ck['state']
        restore_rng(ck['rng'],generator);del ck
        assert frozen_digests(model)==expected_frozen;assert_optimizer(model,opt)
        selected=rd/('selected_epoch_%03d.pt'%state['best_epoch']);assert selected.exists()
        temp=rd/'best.pt.tmp';shutil.copyfile(selected,temp);os.replace(temp,rd/'best.pt')
        (rd/'history.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in state['history']))
    else:
        metrics,arrays=validate(model,data,raw,cfg['batch_size'],thresholds)
        assert abs(metrics['raw_f']['r2']-cfg['epoch_zero_anchor_r2'])<cfg['epoch_zero_anchor_r2_atol']
        assert movement(model,diagnostic_m)['maximum_relative']==0
        state.update(best_sse=metrics['raw_f']['sse'],best_metrics=metrics,epoch_zero_metrics=metrics)
        state['history']=[{'epoch':0,'validation':metrics,'train':None,'adapter_movement':movement(model,diagnostic_m)}]
        save_best(0,metrics,arrays);save_last()
        (rd/'history.jsonl').write_text(json.dumps(state['history'][0],allow_nan=False)+'\n')
    atomic_json({'state':'running','arm':arm,'pid':os.getpid(),'completed_epoch':state['completed_epoch'],
        'manifest_sha256':mh,'last_update_unix':time.time()},rd/'status.json')
    for epoch in range(state['completed_epoch']+1,cfg['epochs']+1):
        if STOP:break
        started=time.time();model.eval();assert_optimizer(model,opt)
        order=generator.permutation(len(indices));global_order=indices[order]
        order_sha=hashlib.sha256(global_order.tobytes()).hexdigest()
        sums={k:0.0 for k in ('total','energy','trace','raw_f')};count=0;clip_count=0;batch_count=0;maxnorm=0.
        for cursor in range(0,len(order),cfg['batch_size']):
            rows=order[cursor:cursor+cfg['batch_size']];idx=indices[rows]
            ir=torch.as_tensor(rows,device='cuda');m={t:a[ir] for t,a in cache.items()}
            y=targets(data,idx);opt.zero_grad(set_to_none=True)
            values=losses(model.from_raw(m),y,data.stats,cfg['train_f_variance'],arm)
            assert torch.isfinite(values['total'])
            values['total'].backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.right_adapter.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
            maxnorm=max(maxnorm,norm);clip_count+=int(norm>cfg['grad_clip']);batch_count+=1
            opt.step();state['steps']+=1
            for k in sums:sums[k]+=float(values[k])*len(idx)
            count+=len(idx)
            if batch_count%200==0:
                atomic_json({'state':'training','arm':arm,'pid':os.getpid(),'epoch':epoch,
                    'completed_epoch':state['completed_epoch'],'cursor':cursor+len(idx),
                    'last_update_unix':time.time(),'manifest_sha256':mh},rd/'status.json')
        assert count==len(indices)
        assert frozen_digests(model)==expected_frozen
        assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('right_adapter.'))
        metrics,arrays=validate(model,data,raw,cfg['batch_size'],thresholds)
        if metrics['raw_f']['sse']<state['best_sse']:
            state.update(best_epoch=epoch,best_sse=metrics['raw_f']['sse'],best_metrics=metrics)
            save_best(epoch,metrics,arrays)
        row={'epoch':epoch,'validation':metrics,'train':{k:v/count for k,v in sums.items()},
            'order_sha256':order_sha,'seconds':time.time()-started,'max_preclip_gradient_norm':maxnorm,
            'gradient_clip_fraction':clip_count/batch_count,'optimizer_batches':batch_count,
            'adapter_movement':movement(model,diagnostic_m),'frozen_parameters_and_buffers_unchanged':True,
            'best_epoch':state['best_epoch'],'best_r2':state['best_metrics']['raw_f']['r2']}
        state['completed_epoch']=epoch;state['history'].append(row);save_last()
        with (rd/'history.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        atomic_json({'state':'epoch_complete','arm':arm,'pid':os.getpid(),'epoch':epoch,
            'completed_epoch':epoch,'last_update_unix':time.time(),'best_epoch':state['best_epoch'],
            'best_r2':state['best_metrics']['raw_f']['r2'],'manifest_sha256':mh},rd/'status.json')
        print(json.dumps({'arm':arm,'epoch':epoch,'native_r2':metrics['raw_f']['r2'],'best_epoch':state['best_epoch'],
            'seconds':row['seconds'],'order_sha256':order_sha,'clip_fraction':row['gradient_clip_fraction'],
            'adapter_movement':row['adapter_movement']}),flush=True)
    if state['completed_epoch']==cfg['epochs']:
        best=torch.load(rd/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(best['model'],strict=True)
        assert frozen_digests(model)==expected_frozen
        final,arrays=validate(model,data,raw,cfg['batch_size'],thresholds)
        assert abs(final['raw_f']['r2']-state['best_metrics']['raw_f']['r2'])<1e-7
        save_arrays(arrays,rd/'best_validation_predictions.npz')
        atomic_json({'epoch':state['best_epoch'],'validation':final},rd/'best_metrics.json')
        receipt={'arm':arm,'manifest_sha256':mh,'completed_epoch':state['completed_epoch'],'selected_epoch':state['best_epoch'],
            'epoch_zero':state['epoch_zero_metrics'],'validation':final,'best_checkpoint':str(rd/'best.pt'),
            'best_checkpoint_sha256':sha(rd/'best.pt'),'last_checkpoint_sha256':sha(rd/'last.pt'),
            'single_checkpoint':True,'test_evaluated':False,'geometry_only_inference':True,
            'frozen_parameters_and_all_buffers_unchanged':True,'cache_receipt_sha256':mf['cache_receipt_sha256']}
        atomic_json(receipt,rd/'FIT_COMPLETE.json');atomic_json({'state':'complete',**receipt},rd/'status.json')
        print(json.dumps({'complete':True,'arm':arm,'selected_epoch':state['best_epoch'],'native_r2':final['raw_f']['r2']}),flush=True)
    else:
        atomic_json({'state':'stopped_resumable','arm':arm,'completed_epoch':state['completed_epoch'],
            'manifest_sha256':mh},rd/'status.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['trace','raw_f']);args=parser.parse_args()
    try:run(args.arm)
    except BaseException as exc:
        rd=ROOT/'runs'/args.arm;rd.mkdir(parents=True,exist_ok=True)
        atomic_json({'arm':args.arm,'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),
            'failed_unix':time.time()},rd/'FAILED.json')
        raise
