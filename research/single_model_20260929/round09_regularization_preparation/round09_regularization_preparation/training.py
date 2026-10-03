"""Matched fixed-budget training; last.pt is the completed-epoch commit point."""
import copy,fcntl,json,math,os,shutil,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,PINS,sha,read,atomic_json
from partition_data import Partition,index_sha
from runtime import TensorData,setup,optimizer,rng_state,restore_rng,atomic_torch
from fresh_model import build_fresh,base_state,tensor_hash,base_loss,decorrelation,TYPES,transport_state
from metrics import evaluate,CONST
from predictor import export_payload
from regularization import selected_decay,optimizer_membership,postclip_diagnostics,verify_roster
STOP=False

def objective_and_diagnostics(model,x,y,stats):
    assert model.adapter_enabled is False
    valid=y['mask_E']&y['mask_A']&y['mask_f']
    assert bool(valid.all()),'Frozen benchmark masks changed; do not exclude rows'
    pred,m,diagnostic=model(**x,return_transport=True);loss=base_loss(pred,y,stats)
    # Auxiliary feature diagnostics have no path into either objective.
    with torch.no_grad():orth=decorrelation(m,valid)
    loss['decorrelation']=orth;loss['base']=loss['total']
    with torch.no_grad():
        balanced=torch.cat([m[t][:,1:].detach().flatten(-2)/math.sqrt(d) for t,d in zip(TYPES,(1,3,5))],-1)
        norms=balanced.norm(dim=-1)[valid]
        diagnostic.update(raw_M_norm_min=float(norms.min()),raw_M_norm_max=float(norms.max()),
                          raw_M_norm_mean=float(norms.mean()))
    return pred,loss,diagnostic


def gradient_norms(model):
    sums={'base':0.,'gate':0.}
    for name,p in model.named_parameters():
        if p.grad is not None:
            key='gate' if '.transport.' in name else 'base'
            sums[key]+=float(p.grad.detach().double().square().sum())
    return {key+'_gradient_l2':value**.5 for key,value in sums.items()}

def stop(*_):
    global STOP
    STOP=True

def atomic_copy(source,target):
    target=Path(target);tmp=target.with_name(target.name+'.tmp')
    with open(source,'rb') as inp,tmp.open('wb') as out:
        shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
    os.replace(tmp,target)
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_arrays(values,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as out:np.savez(out,**values);out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def sync_best(rd,state):
    entry=state['best'];source=rd/entry['checkpoint']
    assert sha(source)==entry['checkpoint_sha256']
    assert sha(rd/entry['predictions'])==entry['predictions_sha256']
    atomic_copy(source,rd/'best.pt');atomic_json(entry,rd/'BEST.json')

def run(arm,permit):
    rd=ROOT/'runs'/arm
    assert not (rd/'FIT_COMPLETE.json').exists(),'Completed arm must never be rerun'
    rd.mkdir(parents=True,exist_ok=True)
    lock=(rd/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:_run(arm,permit,rd)
    except BaseException as exc:
        atomic_json({'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),
            'failed_unix':time.time(),'manifest_sha256':permit['manifest_sha256']},rd/'FAILED.json')
        raise

def _run(arm,permit,rd):
    cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json')
    mh=permit['manifest_sha256'];manifest=permit['manifest'];spec=cfg['arms'][arm]
    assert cfg['epochs']==60 and cfg['scheduler'] is None and cfg['test_access'] is False
    assert not (rd/'FIT_COMPLETE.json').exists(),'Completed arm must never be rerun'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    setup(cfg)
    train=TensorData(Partition('train'),'cuda')
    val=TensorData(Partition('val',validation_authorized=True),'cuda')
    assert len(train)==cfg['train_molecules'] and len(val)==cfg['validation_molecules']
    assert stats['train_indices_sha256']==index_sha(train.global_indices)
    model=build_fresh(mc,stats,spec['mode']).cuda()
    initial=tensor_hash(base_state(model));full_initial=tensor_hash(model.state_dict())
    assert initial==manifest['initial_base_tensor_sha256']
    assert full_initial==manifest['initial_full_tensor_sha256']
    assert spec['mode']=='original'
    roster_sha=verify_roster(model);assert roster_sha==manifest['parameter_roster_sha256']
    opt=optimizer(model,cfg,arm);setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
    state={'completed_epoch':0,'steps':0,'history':[],'best':None}
    atomic_json({'train':train.decode_audit,'validation':val.decode_audit,'test_numeric_rows':0},rd/'DATA_ACCESS.json')
    (rd/'selected').mkdir(exist_ok=True)

    def payload():
        return {'format':'round09_resumable_v1','model':model.state_dict(),'optimizer':opt.state_dict(),
            'parameter_roster_sha256':roster_sha,'weight_decay':selected_decay(cfg,arm),
            'rng':rng_state(generator),'state':state,'config':cfg,'model_config':mc,'stats':stats,
            'arm':arm,'manifest_sha256':mh,'split_manifest_sha256':PINS['SPLIT_MANIFEST.json'],
            'initial_base_tensor_sha256':initial,'initial_full_tensor_sha256':full_initial,
            'train_indices_sha256':index_sha(train.global_indices),'permit':{k:v for k,v in permit.items() if k!='manifest'}}

    def commit(row,arrays):
        nonlocal state
        epoch=row['epoch'];improve=state['best'] is None or row['validation']['pooled']['sse']<state['best']['sse']
        state['completed_epoch']=epoch;state['history'].append(row)
        # Unique immutable versions survive interruption between version and last
        # writes. Only the version referenced by committed last.pt is selected.
        version=f'{epoch:03d}_{time.time_ns()}'
        predpath=rd/'selected'/f'predictions_{version}.npz'
        atomic_arrays(arrays,predpath)
        if improve:
            ckpath=rd/'selected'/f'epoch_{version}.pt'
            entry={'epoch':epoch,'sse':row['validation']['pooled']['sse'],
                'r2':row['validation']['pooled']['r2'],'checkpoint':str(ckpath.relative_to(rd)),
                'predictions':str(predpath.relative_to(rd)),'predictions_sha256':sha(predpath)}
            # Selected checkpoint contains full optimizer/RNG/order/history state.
            state['best']=dict(entry)
            atomic_torch(payload(),ckpath)
            state['best']['checkpoint_sha256']=sha(ckpath)
        state['last_predictions']={'path':str(predpath.relative_to(rd)),'sha256':sha(predpath)}
        atomic_torch(payload(),rd/'last.pt')
        sync_best(rd,state)
        (rd/'history.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in state['history']))
        atomic_json({'state':'epoch_complete','completed_epoch':epoch,'pid':os.getpid(),
            'best':state['best'],'manifest_sha256':mh,'last_update_unix':time.time()},rd/'status.json')
        print(json.dumps(row,allow_nan=False),flush=True)

    if (rd/'last.pt').exists():
        ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
        for key,value in (('config',cfg),('model_config',mc),('stats',stats),('arm',arm),('manifest_sha256',mh),
                          ('initial_base_tensor_sha256',initial),('initial_full_tensor_sha256',full_initial)):
            assert ck[key]==value,('Resume mismatch',key)
        assert ck['train_indices_sha256']==index_sha(train.global_indices)
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);state=ck['state']
        assert ck['parameter_roster_sha256']==roster_sha and ck['weight_decay']==selected_decay(cfg,arm)
        optimizer_membership(model,opt,cfg,arm,require_state=state['steps']>0)
        assert [row['epoch'] for row in state['history']]==list(range(state['completed_epoch']+1))
        restore_rng(ck['rng'],generator);del ck
        assert sha(rd/state['last_predictions']['path'])==state['last_predictions']['sha256']
        sync_best(rd,state)
        (rd/'history.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in state['history']))
        if (rd/'FAILED.json').exists():os.replace(rd/'FAILED.json',rd/f'RESUMED_FAILURE_{time.time_ns()}.json')
    else:
        result,arrays=evaluate(model,val,stats,cfg['batch_size']);assert result['pooled']['count']==66860
        commit({'epoch':0,'validation':result,'order_sha256':None,'seconds':0,'train':None},arrays)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for epoch in range(state['completed_epoch']+1,cfg['epochs']+1):
        if STOP:break
        start=time.time();model.train();order=generator.permutation(len(train));order_hash=index_sha(train.global_indices[order])
        assert order_hash==manifest['epoch_order_sha256'][epoch-1]
        sums=np.zeros(6,dtype=np.float64);count=0;clips=0;batches=0;max_norm=0.
        diag_sums={};diag_min={};diag_max={};grad_sums={'base_gradient_l2':0.,'gate_gradient_l2':0.}
        transport_sums={};regularization_sums={};regularization_min={};regularization_max={}
        gate_start={k:v.detach().clone() for k,v in transport_state(model).items()}
        for cursor in range(0,len(order),cfg['batch_size']):
            if STOP:break
            rows=order[cursor:cursor+cfg['batch_size']];x,y=train.batch(rows);opt.zero_grad(set_to_none=True)
            pred,loss,diagnostic=objective_and_diagnostics(model,x,y,stats)
            assert all(torch.isfinite(v) for v in loss.values())
            assert all(np.isfinite(v) for v in diagnostic.values())
            loss['total'].backward();branch_norms=gradient_norms(model)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
            reg=postclip_diagnostics(model,selected_decay(cfg,arm))
            opt.step();state['steps']+=1;count+=len(rows);batches+=1;clips+=int(norm>cfg['grad_clip']);max_norm=max(max_norm,norm)
            for key,value in reg.items():
                if key.endswith('_min') or key=='radial_beta_min_abs':regularization_min[key]=min(regularization_min.get(key,float('inf')),value)
                elif key.endswith('_max'):regularization_max[key]=max(regularization_max.get(key,-float('inf')),value)
                elif key in ('task_norm_zero','task_norm_below_floor'):regularization_sums[key]=regularization_sums.get(key,0)+value
                else:regularization_sums[key]=regularization_sums.get(key,0.)+value*len(rows)
            p=pred[0].detach().double()*pred[1].detach().double().diagonal(dim1=-2,dim2=-1).sum(-1)*CONST
            fmse=float((p-y['f'].double()).square().mean())
            sums+=np.array([float(loss[k]) for k in ('total','base','energy','trace','decorrelation')]+[fmse])*len(rows)
            for key,value in diagnostic.items():
                if key.endswith('_min'):diag_min[key]=min(diag_min.get(key,float('inf')),value)
                elif key.endswith('_max'):diag_max[key]=max(diag_max.get(key,-float('inf')),value)
                elif key.endswith('_mean'):diag_sums[key]=diag_sums.get(key,0.)+value*len(rows)
                elif key.endswith(('_sum','_count')):transport_sums[key]=transport_sums.get(key,0)+value
            for key,value in branch_norms.items():grad_sums[key]+=value*len(rows)
            if batches%200==0:
                atomic_json({'state':'training','pid':os.getpid(),'epoch':epoch,'completed_epoch':state['completed_epoch'],
                    'cursor':count,'last_update_unix':time.time(),'manifest_sha256':mh},rd/'status.json')
        if STOP:break
        assert count==len(train) and opt.param_groups[0]['lr']==cfg['lr']
        result,arrays=evaluate(model,val,stats,cfg['batch_size']);assert result['pooled']['count']==66860
        movement=sum(float((v-gate_start[k]).double().square().sum()) for k,v in transport_state(model).items())**.5
        diagnostics={**diag_min,**diag_max,**{k:v/count for k,v in diag_sums.items()},
            **{k:v/count for k,v in grad_sums.items()},**transport_sums}
        for key,value in transport_sums.items():
            if key.endswith('_sum'):
                number=transport_sums[key[:-4]+'_count'];assert number>0
                diagnostics[key[:-4]+'_mean']=value/number
        row={'epoch':epoch,'validation':result,'order_sha256':order_hash,'seconds':time.time()-start,
            'train':dict(zip(('total','base','energy','trace','decorrelation','raw_f_mse'),(sums/count).tolist())),
            'transport_diagnostics':diagnostics,'optimizer_batches':batches,'gradient_clip_fraction':clips/batches,
            'max_preclip_gradient_norm':max_norm,'gate_parameter_movement_l2':movement,'learning_rate':cfg['lr']}
        row['regularization_diagnostics']={**regularization_min,**regularization_max,
            **{k:(v if k in ('task_norm_zero','task_norm_below_floor') else v/count) for k,v in regularization_sums.items()}}
        row['weight_decay']=selected_decay(cfg,arm)
        optimizer_membership(model,opt,cfg,arm,require_state=True)
        commit(row,arrays)
    if state['completed_epoch']==cfg['epochs']:
        best=torch.load(rd/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best['model'],strict=True)
        atomic_torch(export_payload(model,mc,stats,{'arm':arm,'epoch':state['best']['epoch'],'manifest_sha256':mh,
            'source_checkpoint_sha256':sha(rd/'best.pt'),'selection':cfg['selection'],
            'optimizer':'Adam_AMSGrad_coupled_postclip','weight_decay':selected_decay(cfg,arm),
            'parameter_roster_sha256':roster_sha}),rd/'geometry_best.pt')
        receipt={'completed_epoch':60,'arm':arm,'manifest_sha256':mh,'steps':state['steps'],
            'initial_base_tensor_sha256':initial,'best':state['best'],'history_sha256':sha(rd/'history.jsonl'),
            'best_checkpoint_sha256':sha(rd/'best.pt'),'last_checkpoint_sha256':sha(rd/'last.pt'),
            'geometry_checkpoint_sha256':sha(rd/'geometry_best.pt'),'test_evaluated':False,
            'source_split_manifest_sha256':PINS['SPLIT_MANIFEST.json']}
        atomic_json(receipt,rd/'FIT_COMPLETE.json');atomic_json({'state':'complete',**receipt},rd/'status.json')
    else:
        atomic_json({'state':'stopped_resumable','completed_epoch':state['completed_epoch'],
            'resume_policy':'Replay incomplete epoch from preceding committed last.pt','manifest_sha256':mh},rd/'status.json')
