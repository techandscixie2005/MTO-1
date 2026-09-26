"""Official scalar DetaNet/MSE; published settings first, documented fallbacks only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,fcntl,hashlib,json,math,random,signal,time,traceback
import numpy as np
import torch
from dataset import Data,ROOT
from detanet_adapter import build_detanet,ef_loss
from objective import spectrum
from trainer import atomic_json,atomic_save,setup,state_hash
STOP=False
def stop(*_):
    global STOP
    STOP=True
def fingerprint():
    files=[ROOT/p for p in ('trainer_detanet.py','detanet_adapter.py','dataset.py','objective.py','trainer.py',
        'data/hashes.json','reports/data_audit.json','reports/detanet_confirmation.json')]
    files+=sorted((ROOT/'official_detanet').rglob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def optimizer_scheduler(model,cfg):
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],amsgrad=True,weight_decay=cfg['weight_decay'])
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=cfg['lr_factor'],patience=cfg['plateau_patience'],
        threshold=cfg['plateau_threshold'],threshold_mode=cfg['plateau_threshold_mode'],min_lr=cfg['min_lr'])
    return opt,sch
@torch.no_grad()
def components(output,y):
    me=y['mask_E'];mf=y['mask_f']
    se=float((output[:,:10][me]-y['E'][me]).double().square().sum())
    sf=float((output[:,10:][mf]-y['f'][mf]).double().square().sum())
    return [se,sf],[int(me.sum()),int(mf.sum())]
def means(sums,counts):
    return [sum(sums)/sum(counts),sums[0]/counts[0],sums[1]/counts[1]]
@torch.no_grad()
def validate(model,data,cfg):
    model.eval();sums=np.zeros(2);counts=np.zeros(2);spec=np.zeros(2);points=0
    idx=data.parts['val']
    for start in range(0,len(idx),cfg['batch_size']):
        x,y=data.batch(idx[start:start+cfg['batch_size']]);p=model(**x);s,c=components(p,y);sums+=s;counts+=c
        mask=y['mask_E']&y['mask_f'];d=(spectrum(p[:,:10],p[:,10:],mask)-spectrum(y['E'],y['f'],mask)).double()
        spec+=[float(d.square().sum()),float(d.abs().sum())];points+=d.numel()
    return means(sums,counts),dict(MSE=spec[0]/points,MAE=spec[1]/points)
def snapshot(model,opt,sch,state,cfg,fp,generator):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),state=state,
        config=cfg,fingerprint=fp,rng_numpy=generator.bit_generator.state,rng_numpy_global=np.random.get_state(),
        rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),rng_python=random.getstate())
def restore(ck,model,opt,sch,generator,cfg,fp):
    assert ck['config']==cfg and ck['fingerprint']==fp,'Resume configuration/source mismatch'
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
    generator.bit_generator.state=ck['rng_numpy'];np.random.set_state(ck['rng_numpy_global']);random.setstate(ck['rng_python'])
    torch.set_rng_state(ck['rng_torch'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
    return ck['state']
def run(name):
    cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text());out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'FIT_COMPLETE.json').exists():return
    assert json.loads((ROOT/cfg['authorization_record']).read_text())['status']=='approved'
    assert json.loads((ROOT/'reports/detanet_training_checks.json').read_text())['passed']
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    setup(cfg['seed'],cfg['num_threads']);data=Data();model=build_detanet(cfg).cuda();initial_hash=state_hash(model.state_dict())
    expected=json.loads((ROOT/'reports/detanet_interface_checks.json').read_text())['initial_state_sha256'];assert initial_hash==expected
    opt,sch=optimizer_scheduler(model,cfg);generator=np.random.default_rng(cfg['seed']);fp=fingerprint()
    state=dict(epoch=1,cursor=0,order=None,steps=0,best=None,best_epoch=0,best_step=0,total_seconds=0.,
        interval_sums=[0.,0.],interval_counts=[0,0],history=[],orders=[],stop_reason=None)
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False);state=restore(ck,model,opt,sch,generator,cfg,fp)
        (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in state['history']))
        print('RESUME',state['epoch'],state['cursor'],state['steps'],flush=True)
    atomic_json(dict(config=cfg,fingerprint=fp,parameters=dict(total=sum(p.numel() for p in model.parameters())),
        initial_state_sha256=initial_hash,gpu=torch.cuda.get_device_name(),pid=os.getpid(),
        no_normalization_for_Ef=True,official_source_unchanged=True),out/'run_manifest.json')
    last=time.monotonic();tick=last
    def checkpoint():
        nonlocal last,tick
        now=time.monotonic();state['total_seconds']+=now-tick;tick=now
        atomic_save(snapshot(model,opt,sch,state,cfg,fp,generator),out/'last.pt');last=time.monotonic()
    checkpoint()
    while state['steps']<cfg['max_steps'] and state['stop_reason'] is None:
        if state['order'] is None:
            state['order']=generator.permutation(data.parts['train']);state['cursor']=0
            state['orders'].append(dict(epoch=state['epoch'],order_sha256=hashlib.sha256(state['order'].tobytes()).hexdigest()))
        order=state['order'];idx=order[state['cursor']:state['cursor']+cfg['batch_size']]
        model.train();x,y=data.batch(idx);opt.zero_grad(set_to_none=True);p=model(**x);loss=ef_loss(p,y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite raw E-f loss')
        loss.backward()
        # Native training has no gradient transform or output correction.
        grad_norm=torch.sqrt(sum(q.grad.detach().square().sum() for q in model.parameters() if q.grad is not None))
        if not torch.isfinite(grad_norm):raise FloatingPointError('Nonfinite gradient norm')
        opt.step();ss,cc=components(p.detach(),y)
        state['interval_sums']=[a+b for a,b in zip(state['interval_sums'],ss)]
        state['interval_counts']=[a+b for a,b in zip(state['interval_counts'],cc)]
        state['steps']+=1;state['cursor']+=len(idx)
        epoch_coord=state['epoch']-1+state['cursor']/len(order)
        full_epoch=state['cursor']>=len(order)
        if state['steps']%cfg['validate_every_steps']==0 or state['steps']==cfg['max_steps']:
            train=means(state['interval_sums'],state['interval_counts']);val,spec=validate(model,data,cfg)
            if not all(math.isfinite(v) for v in train+val+list(spec.values())):raise FloatingPointError('Nonfinite validation')
            if state['best'] is None or val[0]<state['best']:
                state.update(best=val[0],best_epoch=epoch_coord,best_step=state['steps'])
                atomic_save(dict(model=model.state_dict(),epoch=epoch_coord,steps=state['steps'],val=val,val_spectrum=spec,config=cfg,fingerprint=fp),out/'best.pt')
            sch.step(val[0])
            row=dict(epoch=epoch_coord,steps=state['steps'],train=train,val=val,val_spectrum=spec,lr=opt.param_groups[0]['lr'],
                best_val=state['best'],best_epoch=state['best_epoch'],best_step=state['best_step'],grad_norm=float(grad_norm),time=time.time())
            state['history'].append(row);state['interval_sums']=[0.,0.];state['interval_counts']=[0,0]
            if train[0]<=cfg['stop_loss'] and val[0]<=cfg['stop_loss']:state['stop_reason']='published_loss_threshold'
            atomic_json(dict(event='VALIDATION_COMPLETE',**row),out/'status.json');print(json.dumps(row),flush=True)
        else:
            row=None
            if state['steps']%10==0:
                atomic_json(dict(event='TRAIN',epoch=epoch_coord,steps=state['steps'],cursor=state['cursor'],loss=float(loss),
                    lr=opt.param_groups[0]['lr'],time=time.time()),out/'status.json')
        if full_epoch:state['epoch']+=1;state['cursor']=0;state['order']=None
        if row is not None or full_epoch or STOP or time.monotonic()-last>=cfg['checkpoint_seconds']:
            checkpoint()
            if row is not None:
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if full_epoch:atomic_json(state['orders'],out/'epoch_orders.json')
        if STOP:
            atomic_json(dict(event='STOPPED_CHECKPOINTED',steps=state['steps'],time=time.time()),out/'status.json');return
    state['stop_reason']=state['stop_reason'] or 'published_1000000_updates_limit';checkpoint()
    atomic_json(dict(event='FIT_COMPLETE',steps=state['steps'],epochs=state['epoch']-1+state['cursor']/len(data.parts['train']),
        best_epoch=state['best_epoch'],best_step=state['best_step'],best_val=state['best'],seconds=state['total_seconds'],
        reason=state['stop_reason'],time=time.time()),out/'FIT_COMPLETE.json')
    print('FIT_COMPLETE',name,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
    try:run(args.name)
    except BaseException as exc:
        out=ROOT/'runs'/args.name;out.mkdir(exist_ok=True)
        atomic_json(dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),time=time.time()),out/'FAILED.json');raise
