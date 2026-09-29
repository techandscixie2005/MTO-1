#!/usr/bin/env python3
"""Train-only teacher initialization; never evaluates validation or test."""
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from model import ROOT,SRC,ArchitectureMTO,C_F
from dataset import Data

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def mse(model,h,e,tf,var):
    with torch.no_grad():
        f=model.scalar_strength(h)
        if model.arm=='independent_trace':f=C_F*e*f
        return float((f-tf).square().mean()/var)
def main(device):
    cfg=json.loads((ROOT/'config.json').read_text());warm=cfg['warmup']
    assert sha(SRC/'runs/mto_eta0/best.pt')==cfg['source_checkpoint_sha256']
    if (ROOT/'PREPARATION.json').exists():raise RuntimeError('Initialization exists; preserve it rather than overwrite')
    pinned=json.loads((SRC/'data/hashes.json').read_text())
    for name,digest in pinned.items():assert sha(SRC/'data'/name)==digest
    torch.set_num_threads(cfg['num_threads']);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(device=device);train=data.parts['train']
    with np.load(SRC/'data/raw_labels.npz') as r:
        mf=r['mask_f'][train].astype(bool);ma=r['mask_A'][train].astype(bool)
        ft=r['f'][train].astype('float64')[mf]
        st=np.trace(r['A'][train].astype('float64'),axis1=-2,axis2=-1)[ma]
    stats=dict(data.stats,f_mean_train=float(ft.mean()),f_var_train=float(ft.var()),f_std_train=float(ft.std()),trace_mean_train=float(st.mean()),trace_std_train=float(st.std()),normalization_scope='f and trace moments pooled over valid train-only molecule/state entries; population variance')
    assert stats['f_var_train']>0 and stats['trace_std_train']>0
    (ROOT/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    subset=np.random.default_rng(warm['subset_seed']).permutation(train)[:warm['subset_molecules']]
    assert len(set(subset.tolist()))==warm['subset_molecules'] and np.isin(subset,train).all()
    (ROOT/'warmup_subset.json').write_text(json.dumps(dict(indices=subset.tolist(),molecule_ids=data.ids[subset].tolist(),seed=warm['subset_seed'],source='frozen training split only'),indent=2)+'\n')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True);initial=ROOT/'initial';initial.mkdir(exist_ok=True)
    feature_path=cache/'teacher_features.pt'
    if feature_path.exists():
        saved=torch.load(feature_path,map_location='cpu',weights_only=False)
        assert np.array_equal(saved['indices'],subset)
        h,e,tf,ts=(saved[k] for k in ('hidden','E','f','trace'))
        extraction_seconds=0.0
        del data
    else:
        teacher=ArchitectureMTO('original',stats).to(device).eval();features=[];energy=[];f=[];trace=[];start=time.time()
        with torch.no_grad():
            for offset in range(0,len(subset),cfg['batch_size']):
                x,_=data.batch(subset[offset:offset+cfg['batch_size']])
                h,t,e=teacher.features(**x);a=teacher.original_matrix(h,t);ss=a.diagonal(dim1=-2,dim2=-1).sum(-1)
                features.append(h.cpu());energy.append(e.cpu());trace.append(ss.cpu());f.append((C_F*e*ss).cpu())
                if offset%(cfg['batch_size']*32)==0:print(json.dumps({'phase':'teacher_features','molecules':min(offset+cfg['batch_size'],len(subset)),'total':len(subset)}),flush=True)
        h=torch.cat(features).reshape(-1,features[0].shape[-1]);e=torch.cat(energy).reshape(-1);tf=torch.cat(f).reshape(-1);ts=torch.cat(trace).reshape(-1)
        assert torch.isfinite(h).all() and torch.isfinite(tf).all() and (tf>=0).all()
        torch.save(dict(indices=subset,hidden=h,E=e,f=tf,trace=ts),cache/'teacher_features.pt')
        extraction_seconds=time.time()-start;del teacher,data
    results={}
    for arm in cfg['arms']:
        model=ArchitectureMTO(arm,stats);counts=model.parameter_counts();start=time.time()
        if arm in ('original','retained_residual'):
            result=dict(head_warmup_steps=0,teacher_native_f_normalized_MSE_initial=0.,teacher_native_f_normalized_MSE_final=0.,note='Source is its own teacher; all arms reuse same training subset/exposure, no extra labels; do not waste compute optimizing zero self-distillation loss')
        else:
            before=mse(model,h,e,tf,stats['f_var_train'])
            optimizer=torch.optim.Adam(model.strength_head.parameters(),lr=warm['lr'],amsgrad=True)
            gen=torch.Generator().manual_seed(warm['seed']);order_hash=hashlib.sha256()
            for step in range(warm['steps']):
                ix=torch.randint(len(h),(warm['state_batch_size'],),generator=gen)
                order_hash.update(ix.numpy().tobytes());optimizer.zero_grad(set_to_none=True)
                fp=model.scalar_strength(h[ix])
                if arm=='independent_trace':fp=C_F*e[ix]*fp
                loss=(fp-tf[ix]).square().mean()/stats['f_var_train']
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite distillation')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.strength_head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            after=mse(model,h,e,tf,stats['f_var_train'])
            assert np.isfinite(after) and after<before
            result=dict(head_warmup_steps=warm['steps'],teacher_native_f_normalized_MSE_initial=before,teacher_native_f_normalized_MSE_final=after,head_state_order_sha256=order_hash.hexdigest(),note='Only new scalar head updated; shared pretrained representation unchanged')
        result.update(parameter_counts=counts,head_warmup_seconds=time.time()-start)
        torch.save(dict(arm=arm,model=model.state_dict(),stats=stats,source_checkpoint_sha256=cfg['source_checkpoint_sha256'],initialization=result),initial/f'{arm}.pt')
        result['initial_checkpoint_sha256']=sha(initial/f'{arm}.pt');results[arm]=result
    assert results['direct_f']['head_state_order_sha256']==results['independent_trace']['head_state_order_sha256']
    report=dict(status='prepared_no_joint_training',source_checkpoint_sha256=cfg['source_checkpoint_sha256'],config_sha256=sha(ROOT/'config.json'),stats_sha256=sha(ROOT/'stats.json'),subset_sha256=sha(ROOT/'warmup_subset.json'),teacher_feature_seconds=extraction_seconds,source_device=device,feature_shape=list(h.shape),arms=results,source_hashes={str(q):sha(q) for q in [ROOT/'prepare.py',ROOT/'model.py',ROOT/'objective.py',SRC/'data/dataset.npz',SRC/'data/raw_labels.npz']})
    (ROOT/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args();main(args.device)
