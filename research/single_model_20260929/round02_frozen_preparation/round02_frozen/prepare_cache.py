"""Generate a server-only TRAIN raw-M cache under exclusive healthy-GPU lock."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha,current_hashes
from launch import admit,EXPECTED

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);args=p.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text())
    parent_cfg=json.loads((PARENT/'round_config.json').read_text())
    frozen=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())
    assert current_hashes(parent_cfg)==frozen['source_hashes']
    assert args.gpu==cfg['gpu_preparation']
    lock=open('/tmp/mto_pouter_gpu_%d.lock'%args.gpu,'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    xml=admit(args.gpu)
    (ROOT/'CACHE_GPU_ADMISSION.xml').write_text(xml)
    os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[args.gpu]
    import torch
    from model_frozen import SOURCE,TYPES,build,frozen_digests
    from train import setup
    from dataset import Data
    setup(cfg)
    data=Data('cuda');train=data.parts['train'];cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    assert not (ROOT/'CACHE_COMPLETE.json').exists(),'Completed cache already exists'
    assert not list(cache.glob('*.partial')),'Interrupted cache requires explicit recovery'
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        values=z['f'][train][z['mask_f'][train]]
        variance=float(np.var(values,dtype=np.float64))
    assert abs(variance-cfg['train_f_variance'])<1e-15,(variance,cfg['train_f_variance'])
    original=torch.load(parent_cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    sc=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    model=build(sc,data.stats,original['model'],cfg['adapter_seed']).cuda();del original
    before=frozen_digests(model)
    maps={t:np.lib.format.open_memmap(cache/(t+'.npy.partial'),mode='w+',dtype=np.float32,
            shape=(len(train),11,16,d)) for t,d in zip(TYPES,(1,3,5))}
    start=time.time()
    with torch.no_grad():
        for i in range(0,len(train),cfg['cache_batch_size']):
            idx=train[i:i+cfg['cache_batch_size']];x,_=data.batch(idx)
            m=model.raw_states(**x)
            for t in TYPES:
                a=m[t].cpu().numpy();assert np.isfinite(a).all();maps[t][i:i+len(idx)]=a
            if i%(cfg['cache_batch_size']*100)==0:
                print(json.dumps({'cached':i+len(idx),'total':len(train),'seconds':time.time()-start}),flush=True)
    assert frozen_digests(model)==before
    shapes={}
    for t,array in maps.items():
        shapes[t]=list(array.shape);array.flush()
    maps.clear()
    for t in TYPES:os.replace(cache/(t+'.npy.partial'),cache/(t+'.npy'))
    np.save(cache/'indices.npy',train);np.save(cache/'ids.npy',data.ids[train])
    report={'passed':True,'split':'train','molecules':len(train),'shapes':shapes,'dtype':'float32',
        'train_f_population_variance':variance,'cache_batch_size':cfg['cache_batch_size'],
        'source_checkpoint':parent_cfg['source_checkpoint'],'source_checkpoint_sha256':parent_cfg['source_checkpoint_sha256'],
        'parent_manifest_sha256':sha(PARENT/'FROZEN_MANIFEST.json'),
        'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in ('prepare_cache.py','model_frozen.py','config.json')},
        'cache_hashes':{str(path):sha(path) for path in sorted(cache.glob('*.npy'))},
        'frozen_parameter_buffer_hashes':before,'frozen_state_unchanged':True,
        'gpu_uuid':EXPECTED[args.gpu],'seconds':time.time()-start,'data_export_allowed':False,
        'test_inference':False,'fit_performed':False}
    (ROOT/'CACHE_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'cached':len(train),'seconds':report['seconds']}),flush=True)

if __name__=='__main__':main()
