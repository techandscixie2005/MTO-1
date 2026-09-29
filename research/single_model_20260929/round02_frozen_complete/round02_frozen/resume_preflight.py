"""Discarded real-TRAIN F-only updates verify serialized optimizer/RNG resume."""
import fcntl
import io
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha
from launch import admit,EXPECTED

def main():
    cfg=json.loads((ROOT/'config.json').read_text());pc=json.loads((PARENT/'round_config.json').read_text())
    lock=open('/tmp/mto_pouter_gpu_2.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'RESUME_GPU_ADMISSION.xml').write_text(admit(2));os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[2]
    import torch
    from model_frozen import SOURCE,build,losses,frozen_digests,assert_optimizer
    from cache_access import load_cache,targets
    from train import setup,rng_state,restore_rng
    from dataset import Data
    setup(cfg);data=Data('cuda');cache,indices,cr=load_cache(data)
    original=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)
    sc=json.loads((SOURCE/'configs/mto_eta0.json').read_text());results=[]
    def update(model,opt,rows,arm):
        r=torch.as_tensor(rows,device='cuda');m={t:a[r] for t,a in cache.items()}
        opt.zero_grad(set_to_none=True)
        v=losses(model.from_raw(m),targets(data,indices[rows]),data.stats,cfg['train_f_variance'],arm)
        v['total'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.right_adapter.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
        opt.step();return float(norm)
    def same(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    for arm in cfg['arms']:
        pair=[build(sc,data.stats,original['model'],cfg['adapter_seed']).cuda() for _ in range(2)]
        opts=[torch.optim.Adam(m.right_adapter.parameters(),lr=cfg['lr'],amsgrad=True,weight_decay=0) for m in pair]
        gen=np.random.default_rng(cfg['order_seed'])
        first=gen.permutation(len(indices))[:cfg['batch_size']]
        update(pair[0],opts[0],first,arm)
        buf=io.BytesIO();torch.save({'model':pair[0].state_dict(),'optimizer':opts[0].state_dict(),'rng':rng_state(gen)},buf)
        next_rows=gen.permutation(len(indices))[:cfg['batch_size']]
        norm1=update(pair[0],opts[0],next_rows,arm)
        buf.seek(0);saved=torch.load(buf,map_location='cpu',weights_only=False)
        pair[1].load_state_dict(saved['model'],strict=True);opts[1].load_state_dict(saved['optimizer'])
        resumed=np.random.default_rng();restore_rng(saved['rng'],resumed)
        resumed_rows=resumed.permutation(len(indices))[:cfg['batch_size']]
        assert np.array_equal(next_rows,resumed_rows)
        norm2=update(pair[1],opts[1],resumed_rows,arm)
        assert norm1==norm2 and same(pair[0].state_dict(),pair[1].state_dict())
        assert same(opts[0].state_dict(),opts[1].state_dict())
        for m,o in zip(pair,opts):
            assert_optimizer(m,o);assert frozen_digests(m)==cr['frozen_parameter_buffer_hashes']
        results.append({'arm':arm,'next_update_bitwise_equal':True,'optimizer_bitwise_equal':True,
            'next_order_equal':True,'preclip_norm':norm1,'frozen_all_parameters_buffers_unchanged':True})
        del pair,opts,saved,buf
    out={'passed':True,'checks':results,'discarded_updates_only':True,'test_inference':False,
        'source_hashes':{str(ROOT/name):sha(ROOT/name) for name in ('config.json','model_frozen.py','cache_access.py',
            'train_frozen.py','resume_preflight.py')},'cache_receipt_sha256':sha(ROOT/'CACHE_COMPLETE.json')}
    (ROOT/'RESUME_PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
