import json,pathlib
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parent
class Data:
    def __init__(self,device='cuda'):
        self.device=device
        with np.load(ROOT/'data/dataset.npz') as d:
            self.parts={k:d[k].copy() for k in ('train','val','test')};self.ids=d['ids'].copy()
            for k in ('z','pos','E','A','edge'):setattr(self,k,torch.from_numpy(d[k]).to(device))
        with np.load(ROOT/'data/raw_labels.npz') as d:
            assert np.array_equal(d['ids'],self.ids)
            self.f=torch.from_numpy(d['f'].astype('float32')).to(device)
            for k in ('mask_E','mask_A','mask_f'):setattr(self,k,torch.from_numpy(d[k]).to(device))
        # The original sE2/sA2/E_state_mean come from the untouched normalization.json; the new frozen
        # E^2 weight constant is a separate file so normalization.json stays byte-identical to its origin.
        self.stats=json.loads((ROOT/'data/normalization.json').read_text())
        weight=json.loads((ROOT/'data/trace_weight.json').read_text())
        assert float(weight['mean_E2_train'])>0,'mean_E2_train must be positive'
        self.stats['mean_E2_train']=float(weight['mean_E2_train'])
    def batch(self,indices):
        idx=torch.as_tensor(indices,device=self.device,dtype=torch.long);n=len(indices)
        z=self.z[idx];mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.edge[idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x=dict(z=z[mask],pos=self.pos[idx][mask],batch=torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],n=n,edge_index=edge)
        y={k:getattr(self,k)[idx] for k in ('E','A','f','mask_E','mask_A','mask_f')}
        return x,y
