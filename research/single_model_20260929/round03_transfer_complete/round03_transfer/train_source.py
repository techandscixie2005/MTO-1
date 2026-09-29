"""Fixed33 original-MTO fitting on the clean 80% subset; no held-out evaluator."""
import fcntl
import json
import os
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from clean_source import ROOT,SOURCE,FitData,build_random,source_loss,tensor_state_hash
from runtime import atomic_json,atomic_torch,setup,rng_state,restore_rng,optimizer,verify_manifest,sha
STOP=False

def stop(*_):
    global STOP
    STOP=True

def run():
    cfg=json.loads((ROOT/'config.json').read_text());manifest,mh=verify_manifest()
    assert cfg['epochs']==33 and cfg['source_selection']=='fixed_epoch_33' and cfg['scheduler'] is None
    assert cfg['allow_test_evaluation'] is False and cfg['source_outer_validation_evaluation'] is False
    rd=ROOT/'runs/source33';rd.mkdir(parents=True,exist_ok=True)
    lock=(rd/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (rd/'FIT_COMPLETE.json').exists():
        assert json.loads((rd/'FIT_COMPLETE.json').read_text())['manifest_sha256']==mh
        print('Already complete',flush=True);return
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    setup(cfg);data=FitData('cuda');assert len(data.global_indices)==cfg['fit_molecules']
    sc=json.loads((ROOT/'source_config.json').read_text());model=build_random(sc,data.stats).cuda()
    initial_hash=tensor_state_hash(model.state_dict());assert initial_hash==manifest['initial_model_tensor_sha256']
    opt=optimizer(model,cfg);setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
    state={'completed_epoch':0,'steps':0,'history':[]}
    def save_last():
        atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator),
            'state':state,'config':cfg,'source_config':sc,'stats':data.stats,'manifest_sha256':mh,
            'initial_model_tensor_sha256':initial_hash,'fit_index_sha256':data.decode_audit['fit_index_sha256']},rd/'last.pt')
    if (rd/'last.pt').exists():
        ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
        assert ck['config']==cfg and ck['source_config']==sc and ck['stats']==data.stats and ck['manifest_sha256']==mh
        assert ck['fit_index_sha256']==data.decode_audit['fit_index_sha256'] and ck['initial_model_tensor_sha256']==initial_hash
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);state=ck['state']
        restore_rng(ck['rng'],generator);del ck
    else:save_last()
    (rd/'history.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in state['history']))
    if (rd/'FAILED.json').exists():os.replace(rd/'FAILED.json',rd/('RESUMED_FAILURE_%d.json'%time.time_ns()))
    atomic_json(data.decode_audit,rd/'TRAIN_DATA_ACCESS.json')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for epoch in range(state['completed_epoch']+1,cfg['epochs']+1):
        if STOP:break
        started=time.time();model.train();order=generator.permutation(len(data.global_indices))
        order_hash=__import__('hashlib').sha256(data.global_indices[order].tobytes()).hexdigest()
        sums=np.zeros(3,dtype=np.float64);count=0;clips=0;batches=0;max_norm=0.
        for cursor in range(0,len(order),cfg['batch_size']):
            if STOP:break
            rows=order[cursor:cursor+cfg['batch_size']];x,y=data.batch(rows);opt.zero_grad(set_to_none=True)
            values=source_loss(model(**x),y,data.stats);assert all(torch.isfinite(v) for v in values)
            values[0].backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
            opt.step();state['steps']+=1;count+=len(rows);batches+=1;clips+=int(norm>cfg['grad_clip']);max_norm=max(max_norm,norm)
            sums+=np.array([float(v) for v in values])*len(rows)
            if batches%200==0:
                atomic_json({'state':'training','pid':os.getpid(),'epoch':epoch,'completed_epoch':state['completed_epoch'],
                    'cursor':count,'fit_molecules':len(order),'last_update_unix':time.time(),'manifest_sha256':mh},rd/'status.json')
        if STOP:break  # Last atomic checkpoint is the preceding complete epoch.
        assert count==len(order)
        row={'epoch':epoch,'train':dict(zip(('total','energy','trace'),(sums/count).tolist())),
             'seconds':time.time()-started,'order_sha256':order_hash,'optimizer_batches':batches,
             'gradient_clip_fraction':clips/batches,'max_preclip_gradient_norm':max_norm,'learning_rate':opt.param_groups[0]['lr']}
        assert row['learning_rate']==cfg['lr'];state['completed_epoch']=epoch;state['history'].append(row);save_last()
        with (rd/'history.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        atomic_json({'state':'epoch_complete','pid':os.getpid(),'completed_epoch':epoch,
            'last_update_unix':time.time(),'manifest_sha256':mh,'train':row['train']},rd/'status.json')
        print(json.dumps(row),flush=True)
    if state['completed_epoch']==cfg['epochs']:
        atomic_torch({'model':model.state_dict(),'source_config':sc,'stats':data.stats,'epoch':33,
            'manifest_sha256':mh,'selection':'fixed_epoch_33','geometry_only_inference':True,
            'buffer_tensor_sha256':tensor_state_hash(dict(model.named_buffers()))},rd/'source_final.pt')
        receipt={'completed_epoch':33,'manifest_sha256':mh,'fixed_epoch_selection':True,
            'checkpoint':str(rd/'source_final.pt'),'checkpoint_sha256':sha(rd/'source_final.pt'),
            'last_checkpoint_sha256':sha(rd/'last.pt'),'initial_model_tensor_sha256':initial_hash,
            'fit_index_sha256':data.decode_audit['fit_index_sha256'],'steps':state['steps'],
            'heldout_or_outer_labels_used_during_fit':False,'test_evaluated':False}
        atomic_json(receipt,rd/'FIT_COMPLETE.json');atomic_json({'state':'complete',**receipt},rd/'status.json')
    else:
        atomic_json({'state':'stopped_resumable','completed_epoch':state['completed_epoch'],
            'resume_policy':'Replay the incomplete epoch from last.pt','manifest_sha256':mh},rd/'status.json')

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        rd=ROOT/'runs/source33';rd.mkdir(parents=True,exist_ok=True)
        atomic_json({'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),'failed_unix':time.time()},rd/'FAILED.json')
        raise
