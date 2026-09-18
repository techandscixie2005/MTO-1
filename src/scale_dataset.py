import json
from pathlib import Path
import numpy as np
import torch


def chosen_mask(root, scale, a):
    if scale == '1k':
        splits = json.loads((Path(root)/'data/splits_1k.json').read_text())
        return np.isin(a['ids'],[i for v in splits.values() for i in v])
    if scale == '10k':
        return a['subset_10k']
    if scale == 'full':
        return np.ones(len(a['ids']),dtype=bool)
    raise ValueError(scale)


def load_scale(root, scale, splits=('train','validation'), limit=None):
    root = Path(root)
    manifest = json.loads((root/'data'/f'manifest_{scale}.json').read_text())
    graphs = []
    with np.load(root/'data/qm9s_full.npz',allow_pickle=False) as a:
        zz,pp,offsets,y = a['z'],a['pos'],a['offsets'],a['spectra']
        ids,identities,split = a['ids'],a['identities'],a['split']
        chosen = chosen_mask(root,scale,a)
        for name in splits:
            mask = a['fresh_test'] if name=='common_fresh_test' else chosen & (split==('train','validation','test').index(name))
            indices = np.flatnonzero(mask)
            if limit is not None: indices=indices[:limit]
            for j in indices:
                lo,hi=offsets[j:j+2]
                pos,z = torch.from_numpy(pp[lo:hi].copy()),torch.from_numpy(zz[lo:hi].copy())
                edges=((torch.cdist(pos,pos)<5.) & ~torch.eye(len(z),dtype=torch.bool)).nonzero().T.contiguous()
                graphs.append({'z':z,'pos':pos,'edge_index':edges,'spectrum':torch.from_numpy(y[j].copy()),
                               'id':int(ids[j]),'identity':str(identities[j]),'split':name,'row':int(j)})
    return graphs,manifest
