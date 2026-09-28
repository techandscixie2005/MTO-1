#!/usr/bin/env python3
"""Only train and validation rows are materialized; original global indices remain stable."""
import json,sys
from pathlib import Path
import numpy as np
import torch
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
class TrainValData:
    def __init__(self,device="cuda"):
        self.device=device
        with np.load(SRC/"data/dataset.npz",allow_pickle=False) as d:
            self.parts={p:d[p].copy() for p in ("train","val")}
            assert len(self.parts["train"])==120355 and len(self.parts["val"])==6686
            assert len(np.intersect1d(*self.parts.values()))==0
            self.indices=np.concatenate((self.parts["train"],self.parts["val"]))
            self.lookup=np.full(len(d["ids"]),-1,dtype=np.int64)
            self.lookup[self.indices]=np.arange(len(self.indices),dtype=np.int64)
            self.ids=d["ids"][self.indices].copy()
            for k in ("z","pos","E","A","edge"):
                setattr(self,k,torch.from_numpy(d[k][self.indices].copy()).to(device))
        with np.load(SRC/"data/raw_labels.npz",allow_pickle=False) as d:
            assert np.array_equal(d["ids"][self.indices],self.ids)
            self.raw_f=d["f"][self.indices].astype(np.float64)
            self.raw_E=d["E"][self.indices].astype(np.float64)
            self.f=torch.from_numpy(d["f"][self.indices].astype(np.float32)).to(device)
            for k in ("mask_E","mask_A","mask_f"):
                setattr(self,k,torch.from_numpy(d[k][self.indices].copy()).to(device))
        self.stats=json.loads((SRC/"data/normalization.json").read_text())
        assert all(getattr(self,k).all().item() for k in ("mask_E","mask_A","mask_f"))
    def batch(self,indices):
        local=self.lookup[np.asarray(indices,dtype=np.int64)]
        assert (local>=0).all(),"Only train/validation global indices are allowed"
        idx=torch.as_tensor(local,device=self.device,dtype=torch.long);n=len(local)
        z=self.z[idx];mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.edge[idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x=dict(z=z[mask],pos=self.pos[idx][mask],
            batch=torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],
            n=n,edge_index=edge)
        y={k:getattr(self,k)[idx] for k in ("E","A","f","mask_E","mask_A","mask_f")}
        return x,y
    def raw_at(self,indices,key):
        assert key in ("f","E")
        local=self.lookup[np.asarray(indices,dtype=np.int64)]
        assert (local>=0).all()
        return (self.raw_f if key=="f" else self.raw_E)[local]
    def ids_at(self,indices):
        local=self.lookup[np.asarray(indices,dtype=np.int64)]
        assert (local>=0).all()
        return self.ids[local]

