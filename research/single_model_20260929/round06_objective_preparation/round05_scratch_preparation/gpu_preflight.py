"""Exactly twelve discarded optimizer updates on first128 new TRAIN rows."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import builtins,contextlib,copy,gc,io,json,math,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from common import ROOT,PINS,sha,read,immutable_json
from fresh_model import build_fresh,build_original,base_state,tensor_hash,base_loss,decorrelation,training_loss,TYPES
from partition_data import Partition,index_sha
from runtime import TensorData,setup,optimizer,atomic_torch,rng_state,restore_rng
from predictor import load_predictor,export_payload

@contextlib.contextmanager
def forbid_external_data(allowed):
    allowed=Path(allowed).resolve();opened=[]
    def guard(original):
        def call(path,*args,**kwargs):
            if isinstance(path,(str,os.PathLike)) and Path(path).suffix.lower() in ('.pt','.npz','.npy','.json'):
                assert Path(path).resolve()==allowed,('External fixture load',str(path));opened.append(str(path))
            return original(path,*args,**kwargs)
        return call
    with patch('builtins.open',guard(builtins.open)),patch('io.open',guard(io.open)):
        yield
    assert opened

def norm(grads):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))

def compare(left,right,report):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype
        a=left.detach().cpu();b=right.detach().cpu()
        if a.is_floating_point():
            assert torch.allclose(a,b,atol=2e-6,rtol=1e-5)
            if a.numel():report.append(float((a-b).abs().max()))
        else:assert torch.equal(a,b)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:compare(left[key],right[key],report)
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):compare(a,b,report)
    else:assert left==right

def exact(left,right):
    if isinstance(left,torch.Tensor):assert left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,np.ndarray):assert left.dtype==right.dtype and np.array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:exact(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right):exact(a,b)
    else:assert left==right

def main():
    start=time.time();review=read(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert review['passed'] and review['scope']=='round05_128_train_12_discarded_updates'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    cpu=read(ROOT/'CPU_PREFLIGHT.json');assert cpu['passed']
    cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1;setup(cfg)
    part=Partition('train');rows=part.indices[:128].copy();data=TensorData(part,'cuda',rows)
    assert len(data)==128 and all(x['numeric_rows']==128 for x in data.decode_audit)
    out=ROOT/'private_preflight';out.mkdir(exist_ok=False)
    immutable_json({'disposable':True,'production_initialization_prohibited':True,'unique_train_molecules':128,
        'index_sha256':index_sha(rows),'id_sha256':index_sha(data.ids),'planned_executed_updates':12},out/'DISPOSABLE.json')
    report={'passed':False,'train_indices_sha256':index_sha(rows),'train_ids_sha256':index_sha(data.ids),
        'unique_train_molecules':128,'actual_decode_ledger':data.decode_audit,'validation_test_numeric_rows':0,
        'executed_optimizer_updates':0,'arms':{},'device_uuid_environment':os.environ['CUDA_VISIBLE_DEVICES'],
        'resume_tolerances':{'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,'loss_atol':1e-6,'loss_rtol':1e-5},
        'preflight_states_are_never_production_initialization':True}
    x2,_=data.batch(np.arange(2));original=build_original(mc,stats).cuda().eval()
    with torch.no_grad():reference=original(**x2)
    del original;torch.cuda.empty_cache();initials=[]
    batches=(np.arange(64),np.arange(64,128))
    for arm,spec in cfg['arms'].items():
        setup(cfg);model=build_fresh(mc,stats,spec['adapter']).cuda()
        initial=tensor_hash(base_state(model));initials.append(initial)
        assert initial==cpu['initial_base_tensor_sha256'] and tensor_hash(model.state_dict())==cpu['initial_full_tensor_sha256']
        model.eval()
        with torch.no_grad():pred=model(**x2)
        assert all(torch.equal(a,b) for a,b in zip(reference,pred))
        model.train();opt=optimizer(model,cfg);setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
        # The fixture follows the frozen first64 then next64, never samples by labels.
        fixture_order=np.arange(128,dtype=np.int64);order_hash=index_sha(rows[fixture_order])
        x,y=data.batch(batches[0]);pred,m=model(**x,return_aux=True)
        base=base_loss(pred,y,stats)['total'];orth=decorrelation(m,y['mask_E']&y['mask_A']&y['mask_f'])
        params=[p for p in model.parameters() if p.requires_grad]
        bg=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
        og=torch.autograd.grad(orth,params,retain_graph=True,allow_unused=True)
        assert all(torch.isfinite(v).all() for v in bg+og if v is not None)
        adapter_grad=0.;gate_grad=0.
        if spec['adapter']:
            adapter_grad=norm(torch.autograd.grad(base,list(model.right_adapter.mix.parameters()),retain_graph=True))
            gate_grad=norm(torch.autograd.grad(base,list(model.right_adapter.gates.parameters()),retain_graph=True))
            assert adapter_grad>0 and gate_grad==0
        assert norm(bg)>0 and norm(og)>0
        raw_norm=torch.cat([m[t][:,1:].flatten(-2)/math.sqrt(d) for t,d in zip(TYPES,(1,3,5))],-1).norm(dim=-1)
        audit={'base_loss':float(base),'decorrelation':float(orth),'base_gradient_l2':norm(bg),'orth_gradient_l2':norm(og),
            'lambda_weighted_gradient_ratio':spec['lambda']*norm(og)/norm(bg),
            'identity_adapter_mix_gradient_l2':adapter_grad,'identity_upstream_gate_gradient_l2':gate_grad,
            'raw_state_norm_min':float(raw_norm.min()),'raw_state_norm_max':float(raw_norm.max())}
        del pred,m,base,orth,bg,og,params,raw_norm
        def step(which):
            model.train();x,y=data.batch(batches[which]);opt.zero_grad(set_to_none=True)
            _,loss=training_loss(model,x,y,stats,spec['lambda']);assert torch.isfinite(loss['total'])
            loss['total'].backward();clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
            value=float(loss['total']);opt.step();report['executed_optimizer_updates']+=1
            return value,clip
        torch.cuda.reset_peak_memory_stats();started=time.time();first=step(0)
        if spec['adapter']:
            x,y=data.batch(batches[0]);p,_=model(**x,return_aux=True);loss=base_loss(p,y,stats)['total']
            active_gate=norm(torch.autograd.grad(loss,list(model.right_adapter.gates.parameters()),allow_unused=True))
            assert active_gate>0;audit['upstream_gate_gradient_l2_after_update1']=active_gate
            del p,loss
        path=out/(arm+'_after_update1.pt')
        atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator),
            'order':fixture_order,'next_cursor':64,'initial_base_tensor_sha256':initial,'disposable':True},path)
        second=step(1);uninterrupted={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        uninterrupted_opt=copy.deepcopy(opt.state_dict());uninterrupted_rng=copy.deepcopy(rng_state(generator))
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['disposable'] and ck['next_cursor']==64 and np.array_equal(ck['order'],fixture_order)
        model=build_fresh(mc,stats,spec['adapter']).cuda();model.load_state_dict(ck['model'],strict=True)
        opt=optimizer(model,cfg);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],generator)
        exact(ck['rng'],rng_state(generator))
        replay=step(1);model_diff=[];optimizer_diff=[]
        exact(uninterrupted_rng,rng_state(generator));assert index_sha(rows[ck['order']])==order_hash
        compare(uninterrupted,model.state_dict(),model_diff);compare(uninterrupted_opt,opt.state_dict(),optimizer_diff)
        assert np.isclose(second[0],replay[0],atol=1e-6,rtol=1e-5)
        torch.cuda.synchronize();elapsed=time.time()-started
        # Same fixture geometry export, no extra target/geometry rows.
        export=out/(arm+'_geometry_fixture.pt');model.eval()
        atomic_torch(export_payload(model,mc,stats,{'disposable':True,'arm':arm}),export)
        with forbid_external_data(export):
            loaded=load_predictor(export,'cuda')
            with torch.no_grad():a=model(**x2);b=loaded(**x2)
        assert all(torch.equal(a,b) for a,b in zip(a,b))
        report['arms'][arm]={'initial_base_tensor_sha256':initial,'fixture_order_sha256':order_hash,
            'updates':3,'gradient_audit':audit,'update_losses':[first[0],second[0],replay[0]],
            'preclip_norms':[first[1],second[1],replay[1]],'model_replay_max_abs':max(model_diff),
            'optimizer_replay_max_abs':max(optimizer_diff),'loss_replay_abs':abs(second[0]-replay[0]),
            'three_updates_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'geometry_checkpoint_load_forward_bitwise':True,'checkpoint_sha256':sha(path)}
        report['arms'][arm]['rng_and_order_exact_before_and_after_replay']=True
        del model,opt,ck,loaded,uninterrupted,uninterrupted_opt,x,y,a,b;gc.collect();torch.cuda.empty_cache()
    assert len(set(initials))==1 and report['executed_optimizer_updates']==12
    report['source_hashes']=review['source_hashes'];report['technical_review_sha256']=sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    report['seconds']=time.time()-start;report['passed']=True
    immutable_json(report,ROOT/'GPU_PREFLIGHT.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
