"""Aggregate train-only diagnostics after the frozen factorial pilot."""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from architecture.model import SOURCE,TYPES,build_model,load_baseline
from metrics import C_F,score
from freeze import sha
sys.path.insert(0,str(SOURCE))
from dataset import Data

ROOT=Path(__file__).resolve().parent
QUANTILES=[0,.01,.1,.5,.9,.99,1]

def subset():
    with np.load(SOURCE/'data/dataset.npz') as z:
        train=z['train'];ids=z['ids']
        selected=np.random.default_rng(20260930).choice(train,256,replace=False)
        record={'subset_split':'train','seed':20260930,'molecules':256,
                'indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
                'molecule_ids_sha256':hashlib.sha256(ids[selected].tobytes()).hexdigest(),
                'training_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest()}
    return selected,record

def describe(value):
    value=np.asarray(value,dtype=np.float64).ravel()
    assert np.isfinite(value).all()
    return {'count':int(value.size),'mean':float(value.mean()),'rms':float(np.sqrt(np.square(value).mean())),
            'quantiles':{str(q):float(v) for q,v in zip(QUANTILES,np.quantile(value,QUANTILES))}}

def balanced(m):
    return torch.cat([m[t][:,1:].flatten(-2)/math.sqrt(d) for t,d in zip(TYPES,(1,3,5))],-1)

@torch.no_grad()
def run_model(model,data,indices):
    model.eval();norms=[];grams=[];relative=[];absolute=[];ee=[];ff=[]
    by_irrep={t:[] for t in TYPES}
    for start in range(0,len(indices),32):
        x,y=data.batch(indices[start:start+32])
        (energy,matrix),raw=model(**x,return_aux=True)
        r=balanced(raw)
        norm=r.norm(dim=-1)
        unit=r/norm.clamp_min(1e-8)[...,None]
        gram=unit@unit.transpose(-1,-2)
        valid=y['mask_E']&y['mask_A']&y['mask_f']
        pair=valid[:,:,None]&valid[:,None,:]&torch.triu(torch.ones(10,10,device=r.device,dtype=torch.bool),diagonal=1)
        grams.append(gram[pair].cpu().numpy())
        norms.append(norm.cpu().numpy())
        right={t:raw[t][:,1:] for t in TYPES}
        transformed=model.right_adapter(right) if model.adapter_enabled else right
        delta=torch.cat([(transformed[t]-right[t]).flatten(-2)/math.sqrt(d)
                         for t,d in zip(TYPES,(1,3,5))],-1).norm(dim=-1)
        absolute.append(delta.cpu().numpy())
        relative.append((delta/norm.clamp_min(1e-8)).cpu().numpy())
        for t in TYPES:
            numer=(transformed[t]-right[t]).flatten(-2).norm(dim=-1)
            denom=right[t].flatten(-2).norm(dim=-1).clamp_min(1e-8)
            by_irrep[t].append((numer/denom).cpu().numpy())
        en=energy.cpu().numpy().astype(np.float64)
        trace=np.trace(matrix.cpu().numpy().astype(np.float64),axis1=-2,axis2=-1)
        ee.append(en);ff.append(C_F*en*trace)
    norms=np.concatenate(norms);grams=np.concatenate(grams)
    relative=np.concatenate(relative);absolute=np.concatenate(absolute)
    result={'raw_offdiagonal_cosine':describe(grams),'raw_offdiagonal_cosine_squared':describe(np.square(grams)),
        'raw_state_norm':describe(norms),'raw_state_norm_per_state':[describe(norms[:,j]) for j in range(10)],
        'adapter_delta_absolute':describe(absolute),'adapter_delta_relative':describe(relative),
        'adapter_delta_relative_per_state':[describe(relative[:,j]) for j in range(10)],
        'adapter_delta_relative_by_irrep':{t:describe(np.concatenate(v)) for t,v in by_irrep.items()}}
    return result,{'energy':np.concatenate(ee),'native_f':np.concatenate(ff)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args()
    indices,record=subset()
    path=ROOT/'MECHANISM_CONFIG.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        assert frozen['subset']==record
        assert frozen['script_sha256']==sha(__file__),'Mechanism audit source changed after subset freeze'
    else:
        frozen={'subset':record,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'script_sha256':sha(__file__),'primary_fit_sources_modified':False,
            'selection_use':False,'checkpoints':['best','last'],'device':'cpu',
            'interpretation':'train-only descriptive feature/adapter/output drift; not validation selection'}
        path.write_text(json.dumps(frozen,indent=2)+'\n')
    if args.freeze_only:
        print(json.dumps(frozen,indent=2));return
    config=json.loads((ROOT/'round_config.json').read_text())
    assert all((ROOT/'runs'/a/'FIT_COMPLETE.json').is_file() for a in config['arms']), 'Wait for terminal pilot'
    torch.set_num_threads(2)
    data=Data('cpu');source_cfg=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    initial=torch.load(config['source_checkpoint'],map_location='cpu',weights_only=False)
    base=load_baseline(build_model(source_cfg,data.stats,False),initial['model'])
    baseline,bpred=run_model(base,data,indices)
    del base,initial
    result={'configuration':frozen,'baseline':baseline,'arms':{},
        'drift_metric_reference':'Epoch0 predictions: drift R2 denotes baseline-output agreement, not label predictive accuracy'}
    for arm,ac in config['arms'].items():
        result['arms'][arm]={}
        for tag in ('best','last'):
            cp=ROOT/'runs'/arm/(tag+'.pt')
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            model=build_model(source_cfg,data.stats,ac['adapter'])
            model.load_state_dict(ck['model'],strict=True)
            value,pred=run_model(model,data,indices)
            value.update(checkpoint_sha256=sha(cp),epoch=ck.get('epoch',ck.get('state',{}).get('completed_epoch')),
                         adapter_enabled=ac['adapter'])
            value['drift_vs_epoch0']={key:{'pooled':score(bpred[key],pred[key]),
                'signed_mean_change':float((pred[key]-bpred[key]).mean()),
                'per_state':[score(bpred[key][:,j],pred[key][:,j]) for j in range(10)]} for key in pred}
            result['arms'][arm][tag]=value
            del model,ck
    (ROOT/'MECHANISM_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Train-only mechanism diagnostics','',
        'Fixed256-molecule training subset; no selection uses these aggregates. Predictions and feature arrays were not written. Full quantiles and per-state drift appear in MECHANISM_RESULTS.json.','',
        '| Arm/checkpoint | Epoch | Mean raw overlap squared | Mean relative adapter change | Native-f drift RMSE | Energy drift RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,checks in result['arms'].items():
        for name,v in checks.items():
            lines.append('| %s/%s | %s | %.6f | %.6g | %.6g | %.6g |'%(arm,name,v['epoch'],
                v['raw_offdiagonal_cosine_squared']['mean'],v['adapter_delta_relative']['mean'],
                v['drift_vs_epoch0']['native_f']['pooled']['rmse'],v['drift_vs_epoch0']['energy']['pooled']['rmse']))
    lines+=['','Baseline mean raw overlap squared: %.6f.'%baseline['raw_offdiagonal_cosine_squared']['mean'],
        'A changed latent overlap or adapter output demonstrates optimization effect, not physical wavefunction decorrelation or a causally identified error source.']
    (ROOT/'MECHANISM_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
