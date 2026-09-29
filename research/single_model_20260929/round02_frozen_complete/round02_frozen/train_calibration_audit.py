"""All-TRAIN native prediction moments; affine coefficients are descriptive only."""
import fcntl
import json
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha
from launch import admit,EXPECTED
from variance_provenance import selected_rows

def describe(moment):
    n,sp,sy,spp,syy,spy,sse=map(float,moment)
    vp=spp-sp*sp/n;vy=syy-sy*sy/n;cov=spy-sp*sy/n
    alpha=cov/vp;beta=sy/n-alpha*sp/n
    return {'count':int(n),'native_sse':sse,'native_mse':sse/n,'native_r2':1-sse/vy,
        'mean_prediction':sp/n,'mean_label':sy/n,'prediction_population_variance':vp/n,
        'label_population_variance':vy/n,'unconstrained_affine_slope':alpha,'unconstrained_affine_intercept':beta,
        'affine_applied_or_scored':False,'moments':{'sum_prediction':sp,'sum_label':sy,
            'sum_prediction_squared':spp,'sum_label_squared':syy,'sum_cross_product':spy}}

def main():
    cfg=json.loads((ROOT/'config.json').read_text());mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    for path,value in mf['source_hashes'].items():assert sha(path)==value
    cr=json.loads((ROOT/'CACHE_COMPLETE.json').read_text())
    for path,value in cr['cache_hashes'].items():assert sha(path)==value
    pc=json.loads((PARENT/'round_config.json').read_text())
    source=Path(pc['source']);indices=np.load(ROOT/'cache/indices.npy');ids=np.load(ROOT/'cache/ids.npy')
    with np.load(source/'data/dataset.npz') as z:
        assert np.array_equal(indices,z['train']) and np.array_equal(ids,z['ids'][indices])
    order=np.argsort(indices);undo=np.argsort(order)
    with zipfile.ZipFile(source/'data/raw_labels.npz') as z:
        truth=selected_rows(z,'f',indices[order])[undo].astype(np.float64)
        mask=selected_rows(z,'mask_f',indices[order])[undo].astype(bool)
    assert mask.sum()==1203550
    lock=open('/tmp/mto_pouter_gpu_1.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'TRAIN_CALIBRATION_GPU_ADMISSION.xml').write_text(admit(1));os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[1]
    import torch
    from model_frozen import TYPES,build,frozen_digests
    from train import setup
    from metrics import C_F,CAL_ALPHA,CAL_BETA
    setup(cfg)
    original=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)
    sc=json.loads((source/'configs/mto_eta0.json').read_text());stats=json.loads((source/'data/normalization.json').read_text())
    model=build(sc,stats,original['model'],cfg['adapter_seed']).cuda();model.requires_grad_(False);del original
    assert frozen_digests(model)==cr['frozen_parameter_buffer_hashes']
    cache={t:torch.from_numpy(np.load(ROOT/'cache'/(t+'.npy'))).cuda() for t in TYPES}
    sums=np.zeros((10,7),dtype=np.float64);start=time.time()
    with torch.no_grad():
        for i in range(0,len(indices),64):
            energy,matrix=model.from_raw({t:a[i:i+64] for t,a in cache.items()})
            energy=energy.cpu().numpy().astype(np.float64);matrix=matrix.cpu().numpy().astype(np.float64)
            pred=C_F*energy*np.trace(matrix,axis1=-2,axis2=-1);y=truth[i:i+64];use=mask[i:i+64]
            for j in range(10):
                p=pred[:,j][use[:,j]];q=y[:,j][use[:,j]]
                sums[j]+=np.array([len(p),p.sum(),q.sum(),np.square(p).sum(),np.square(q).sum(),
                                  (p*q).sum(),np.square(p-q).sum()],dtype=np.float64)
    assert frozen_digests(model)==cr['frozen_parameter_buffer_hashes']
    pooled=describe(sums.sum(axis=0));states=[{'state':j+1,**describe(sums[j])} for j in range(10)]
    result={'passed':True,'split':'all_valid_frozen_TRAIN','molecules':len(indices),'valid_labels':int(mask.sum()),
        'zero_labels':int(np.count_nonzero(truth[mask]==0)),'pooled':pooled,'per_state':states,
        'coefficient_interpretation':'unconstrained least-squares moment diagnostics only; not deployed, not validation scored',
        'historical_fixed_validation_coefficients':{'alpha':CAL_ALPHA,'beta':CAL_BETA,'already_existing':True},
        'pooled_coefficient_differences_train_minus_historical':{'alpha':pooled['unconstrained_affine_slope']-CAL_ALPHA,
            'beta':pooled['unconstrained_affine_intercept']-CAL_BETA},
        'source_checkpoint_sha256':pc['source_checkpoint_sha256'],'cache_receipt_sha256':sha(ROOT/'CACHE_COMPLETE.json'),
        'frozen_manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),'frozen_weights_and_buffers_unchanged':True,
        'decoded_validation_or_test_label_rows':0,'validation_or_test_inference':False,'fit_updates':0,'raw_arrays_exported':False,
        'FP64_accumulation':True,'gpu_uuid':EXPECTED[1],'seconds':time.time()-start,
        'source_hashes':{str(ROOT/name):sha(ROOT/name) for name in ('train_calibration_audit.py','variance_provenance.py','model_frozen.py')},
        'training_index_sha256':__import__('hashlib').sha256(indices.tobytes()).hexdigest()}
    (ROOT/'TRAIN_CALIBRATION_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pooled':pooled,'per_state_slopes':[s['unconstrained_affine_slope'] for s in states],
        'historical_fixed_validation_coefficients':result['historical_fixed_validation_coefficients'],'seconds':result['seconds']},indent=2))

if __name__=='__main__':main()
