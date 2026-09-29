"""Strict cache provenance and global-index-to-row alignment."""
import json
from pathlib import Path
import numpy as np
import torch
from model_frozen import TYPES
from freeze import sha

ROOT=Path(__file__).resolve().parent

def load_cache(data,device='cuda'):
    receipt=json.loads((ROOT/'CACHE_COMPLETE.json').read_text())
    assert receipt['passed'] and receipt['split']=='train' and receipt['frozen_state_unchanged']
    for path,expected in receipt['source_hashes'].items():assert sha(path)==expected,('Cache source changed',path)
    for path,expected in receipt['cache_hashes'].items():assert sha(path)==expected,('Cache changed',path)
    indices=np.load(ROOT/'cache/indices.npy');ids=np.load(ROOT/'cache/ids.npy')
    assert np.array_equal(indices,data.parts['train']) and np.array_equal(ids,data.ids[indices])
    assert len(np.unique(indices))==len(indices)==receipt['molecules']
    arrays={}
    for t in TYPES:
        a=np.load(ROOT/'cache'/(t+'.npy'))
        assert list(a.shape)==receipt['shapes'][t] and a.dtype==np.float32 and np.isfinite(a).all()
        arrays[t]=torch.from_numpy(a).to(device)
    return arrays,indices,receipt

def targets(data,indices):
    idx=torch.as_tensor(indices,dtype=torch.long,device=data.device)
    return {k:getattr(data,k)[idx] for k in ('E','A','f','mask_E','mask_A','mask_f')}
