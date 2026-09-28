import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,hashlib,json,math,pathlib,random,signal,time,traceback
import numpy as np
import torch
from dataset import Data,ROOT
from model_factory import build
from objective import losses,oscillator,spectrum
STOP=False
def stop(*_):
    global STOP
    STOP=True
def atomic_json(obj,path):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)
def atomic_save(obj,path):
    path=pathlib.Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def fingerprint():
    # prepare.py joins the list because it produces the frozen mean_E2_train constant the objective reads.
    paths=[ROOT/k for k in ('trainer.py','model_factory.py','dataset.py','objective.py','evaluate.py','prepare.py','campaign.json','data/hashes.json','data/normalization.json','data/trace_weight.json')]
    paths+=sorted((ROOT/'frozen_reference').rglob('*.py'))
    paths += [ROOT/'reports/initialization.json',ROOT/'reports/data_audit.json',ROOT/'initial/original.pt',ROOT/'initial/p_outer.pt']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def setup(seed,threads=2):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def totals(sums,counts,eta):
    le=sums[0]/max(counts[0],1);ls=sums[1]/max(counts[1],1);lq=sums[2]/max(counts[1],1)
    return [le+ls+lq if eta==1 else ls,le,ls,lq]
@torch.no_grad()
def validate(model,data,indices,cfg,heartbeat=None):
    model.eval();sums=np.zeros(3);counts=np.zeros(2);spec=np.zeros(2);num=0
    for a in range(0,len(indices),cfg['batch_size']):
        idx=indices[a:a+cfg['batch_size']];x,y=data.batch(idx);pred=model(**x)
        vals=losses(pred,y,data.stats,cfg['eta'])
        ne=int(y['mask_E'].sum());na=int(y['mask_A'].sum())
        sums+=np.array([float(v) for v in vals[1:]])*np.array([ne,na,na]);counts+=[ne,na]
        if heartbeat:heartbeat()
    return totals(sums,counts,cfg['eta']),{}
def run(name,preflight_out=None):
    import fcntl
    cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text());out=ROOT/'runs'/name
    if preflight_out is not None:
        out=pathlib.Path(preflight_out).resolve();assert out.is_relative_to((ROOT/'reports').resolve())
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'FIT_COMPLETE.json').exists():print('Already completed');return
    if preflight_out is None and not json.loads((ROOT/'reports/preflight.json').read_text())['passed']:raise RuntimeError('Preflight not passed')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    setup(cfg['seed'],cfg['num_threads']);data=Data();model=build(cfg,data.stats).cuda()
    # Anchor the frozen constant on this machine's data copy, not only on the cross-machine sha256.
    recorded=json.loads((ROOT/'reports/trace_weight.json').read_text())['mean_E2_train']
    assert abs(data.stats['mean_E2_train']-recorded)<1e-9,'mean_E2_train differs from the value recorded at prepare time'
    assert data.stats['mean_E2_train']==float(json.loads((ROOT/'data/trace_weight.json').read_text())['mean_E2_train'])
    init_hash=state_hash(model.state_dict())
    expected=json.loads((ROOT/'reports/initialization.json').read_text())['groups'][name]
    assert init_hash==expected,'Initial parameters differ'
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],amsgrad=True,weight_decay=cfg['weight_decay'])
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=cfg['plateau_patience'],threshold=1e-4,threshold_mode='rel',min_lr=cfg['min_lr'])
    generator=np.random.default_rng(cfg['seed']);fp=fingerprint()
    state=dict(epoch=1,cursor=0,order=None,sums=[0.,0.,0.],counts=[0,0],best=None,best_epoch=0,bad_epochs=0,lr_reductions=0,last_reduction=0,steps=0,total_seconds=0.,history=[])
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        if ck['fingerprint']!=fp or ck['config']!=cfg:raise RuntimeError('Resume fingerprint/config mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);state=ck['state']
        generator.bit_generator.state=ck['rng_numpy'];torch.set_rng_state(ck['rng_torch'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy_global'])
        (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in state['history']))
        print('RESUME',state['epoch'],state['cursor'],flush=True)
    counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for p in model.core.parameters()))
    atomic_json(dict(config=cfg,parameters=counts,fingerprint=fp,normalization=data.stats,gpu=torch.cuda.get_device_name(),pid=os.getpid(),initial_state_sha256=init_hash),out/'run_manifest.json')
    last_checkpoint=time.monotonic();tick=last_checkpoint
    def checkpoint():
        nonlocal last_checkpoint,tick
        now=time.monotonic();state['total_seconds']+=now-tick;tick=now
        atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),state=state,
            config=cfg,fingerprint=fp,rng_numpy=generator.bit_generator.state,rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(),rng_python=random.getstate(),rng_numpy_global=np.random.get_state()),out/'last.pt')
        last_checkpoint=time.monotonic()
    checkpoint();early=False
    def heartbeat():
        if time.monotonic()-last_checkpoint>=min(cfg['checkpoint_seconds'],540):checkpoint()
    while state['epoch']<=cfg['max_epochs']:
        previous_epoch=state['epoch']-1
        early=previous_epoch>=cfg['min_epochs'] and state['bad_epochs']>=cfg['early_patience'] and state['lr_reductions']>=2 and previous_epoch-state['last_reduction']>=50
        if early:break
        epoch=state['epoch'];epoch_start=time.monotonic()
        if state['order'] is None:
            state['order']=generator.permutation(data.parts['train']);state['cursor']=0;state['sums']=[0.,0.,0.];state['counts']=[0,0]
            atomic_json(dict(epoch=epoch,sha256=hashlib.sha256(state['order'].tobytes()).hexdigest()),out/'current_order.json')
        order=state['order'];model.train()
        while state['cursor']<len(order):
            a=state['cursor'];idx=order[a:a+cfg['batch_size']]
            x,y=data.batch(idx);opt.zero_grad(set_to_none=True)
            values=losses(model(**x),y,data.stats,cfg['eta']);total=values[0]
            if not torch.isfinite(total):raise FloatingPointError('nonfinite loss')
            total.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            opt.step();v=[float(q.detach()) for q in values]
            ne=int(y['mask_E'].sum());na=int(y['mask_A'].sum())
            state['sums']=[s+t*c for s,t,c in zip(state['sums'],v[1:],[ne,na,na])]
            state['counts']=[state['counts'][0]+ne,state['counts'][1]+na]
            state['cursor']+=len(idx);state['steps']+=1
            if state['steps']%20==0:
                status=dict(event='TRAIN',epoch=epoch,cursor=state['cursor'],train_count=len(order),steps=state['steps'],loss=v,
                    lr=opt.param_groups[0]['lr'],grad_norm=float(gn),time=time.time(),seconds_this_epoch=time.monotonic()-epoch_start)
                atomic_json(status,out/'status.json');print(json.dumps(status),flush=True)
            if STOP or time.monotonic()-last_checkpoint>=min(cfg['checkpoint_seconds'],540):
                checkpoint()
                if STOP:
                    atomic_json(dict(event='STOPPED_CHECKPOINTED',epoch=epoch,cursor=state['cursor'],time=time.time()),out/'status.json');return
        train=totals(state['sums'],state['counts'],cfg['eta']);val,spec=validate(model,data,data.parts['val'],cfg,heartbeat)
        if not all(math.isfinite(v) for v in val+list(spec.values())):raise FloatingPointError('nonfinite validation')
        improved=state['best'] is None or val[0]<state['best']
        if improved:
            state['best']=val[0];state['best_epoch']=epoch;state['bad_epochs']=0
            atomic_save(dict(model=model.state_dict(),epoch=epoch,val=val,val_spectrum=spec,config=cfg,fingerprint=fp),out/'best.pt')
        else:state['bad_epochs']+=1
        previous_lr=opt.param_groups[0]['lr'];sch.step(val[0])
        if opt.param_groups[0]['lr']<previous_lr:state['lr_reductions']+=1;state['last_reduction']=epoch
        row=dict(epoch=epoch,train=train,val=val,val_spectrum=spec,lr=opt.param_groups[0]['lr'],best_val=state['best'],best_epoch=state['best_epoch'],
            seconds=time.monotonic()-epoch_start,steps=state['steps'],peak_memory_bytes=torch.cuda.max_memory_allocated(),time=time.time(),
            order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
        state['history'].append(row)
        atomic_json(dict(event='EPOCH_COMPLETE',**row),out/'status.json');print(json.dumps(row),flush=True)
        early=epoch>=cfg['min_epochs'] and state['bad_epochs']>=cfg['early_patience'] and state['lr_reductions']>=2 and epoch-state['last_reduction']>=50
        state['epoch']+=1;state['order']=None;state['cursor']=0;checkpoint()
        (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in state['history']))
        if STOP:return
        if early:break
    atomic_json(dict(event='FIT_COMPLETE',epochs=state['epoch']-1,best_epoch=state['best_epoch'],best_val=state['best'],
        seconds=state['total_seconds'],steps=state['steps'],parameters=counts,reason='early_stop' if early else 'max_epochs',time=time.time()),out/'FIT_COMPLETE.json')
    print('FIT_COMPLETE',name,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
    try:run(args.name)
    except BlockingIOError:
        print('Worker already active; duplicate launch refused',flush=True)
    except BaseException as exc:
        path=ROOT/'runs'/args.name;path.mkdir(exist_ok=True)
        atomic_json(dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),time=time.time()),path/'FAILED.json')
        raise
