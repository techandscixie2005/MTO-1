import json,pathlib
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parent
class Data:
    def __init__(self,device='cuda'):
        with np.load(ROOT/'data/dataset.npz') as d:
            self.parts={k:d[k].copy() for k in ('train','val','test')};self.ids=d['ids'].copy()
            # Inputs and targets fit comfortably in each 80 GB card; no repeated NPZ decompression or worker copying.
            self.z=torch.from_numpy(d['z']).to(device)
            self.pos=torch.from_numpy(d['pos']).to(device)
            self.E=torch.from_numpy(d['E']).to(device);self.A=torch.from_numpy(d['A']).to(device)
            self.edge=torch.from_numpy(d['edge']).to(device)
        self.device=device;self.stats=json.loads((ROOT/'data/normalization.json').read_text())
    def batch(self,indices):
        idx=torch.as_tensor(indices,device=self.device,dtype=torch.long);n=len(indices)
        z=self.z[idx];mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.edge[idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x=dict(z=z[mask],pos=self.pos[idx][mask],batch=torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],n=n,edge_index=edge)
        return x,(self.E[idx],self.A[idx])
