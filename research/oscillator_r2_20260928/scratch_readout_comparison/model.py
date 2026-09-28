#!/usr/bin/env python3
"""Fresh matched DetaNet cores with native or MTO direct positive readout."""
import json,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(SRC))
from model_factory import build as build_source
from detanet_adapter import NativeEfDetaNet
from models import unpack,TYPES
def invsoftplus(x):
    t=torch.as_tensor(x,dtype=torch.float64)
    assert torch.isfinite(t).all() and (t>0).all()
    return t+torch.log(-torch.expm1(-t))
class NativeDirect(nn.Module):
    def __init__(self,stats):
        super().__init__()
        torch.manual_seed(11)
        self.base=NativeEfDetaNet()
        self.register_buffer("offset_E",invsoftplus(stats["mu_E"]).float())
        self.register_buffer("offset_f",invsoftplus(
            torch.as_tensor(stats["mu_f"],dtype=torch.float64)/stats["sf"]).float())
        self.register_buffer("sf",torch.tensor(stats["sf"],dtype=torch.float32))
        last=[m for m in self.base.sout.modules() if isinstance(m,nn.Linear)][-1]
        assert last.in_features==128 and last.out_features==20
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    def forward(self,z,pos,batch,n,edge_index=None):
        logits=self.base(z=z,pos=pos,batch=batch,n=n,edge_index=edge_index)
        assert logits.shape==(n,20)
        return {"E":F.softplus(logits[:,:10]+self.offset_E),
                "f":self.sf*F.softplus(logits[:,10:]+self.offset_f)}
class MTODirect(nn.Module):
    def __init__(self,stats):
        super().__init__()
        cfg=json.loads((SRC/"configs/mto_eta0.json").read_text())
        assert cfg["seed"]==11
        self.base=build_source(cfg,stats)
        decoder=self.base.decoder
        assert decoder.energy_offset.shape==(10,)
        del decoder.energy_offset
        decoder.register_buffer("energy_offset",invsoftplus(stats["mu_E"]).float())
        del decoder.beta_head
        del decoder.tensor_gate
        self.strength_head=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1))
        self.register_buffer("offset_f",invsoftplus(
            torch.as_tensor(stats["mu_f"],dtype=torch.float64)/stats["sf"]).float())
        self.register_buffer("sf",torch.tensor(stats["sf"],dtype=torch.float32))
        nn.init.zeros_(decoder.energy_head.weight)
        nn.init.zeros_(decoder.energy_head.bias)
        nn.init.zeros_(self.strength_head[-1].weight)
        nn.init.zeros_(self.strength_head[-1].bias)
    def features(self,z,pos,batch,n,edge_index=None):
        s,t=self.base.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        atom=unpack(t,self.base.tensor_irreps)
        atom["0e"]=s[...,None]
        m,_=self.base.mto({k:atom[k] for k in TYPES},z,batch,n,False)
        scalar,_,_=self.base.cg(m)
        h=self.base.decoder.trunk(scalar)
        return h,s,t
    def forward(self,z,pos,batch,n,edge_index=None):
        h,_,_=self.features(z,pos,batch,n,edge_index)
        E=F.softplus(self.base.decoder.energy_head(h).squeeze(-1)+
                     self.base.decoder.energy_offset)
        f=self.sf*F.softplus(self.strength_head(h).squeeze(-1)+self.offset_f)
        return {"E":E,"f":f}
def make_pair(stats):
    """One fresh MTO core is copied by semantic module key to native; no checkpoint load."""
    mto=MTODirect(stats)
    native=NativeDirect(stats)
    core_state=mto.base.core.state_dict()
    target=native.base.state_dict()
    mapping={}
    for key,value in core_state.items():
        assert key.startswith(("Embedding.","Radial.","blocks.")),key
        assert key in target and target[key].shape==value.shape,(key,value.shape,target.get(key))
        mapping[key]=key
        target[key]=value.detach().clone()
    native.base.load_state_dict(target)
    for key,value in core_state.items():
        assert torch.equal(native.base.state_dict()[key],value),key
    assert torch.equal(native.base.mass,mto.base.core.mass)
    assert torch.equal(native.base.Embedding.elec,mto.base.core.Embedding.elec)
    return {"native":native,"mto":mto},mapping
def counts(model):
    core=(model.base.core.parameters() if isinstance(model,MTODirect) else
          (p for k,p in model.base.named_parameters()
           if k.startswith(("Embedding.","Radial.","blocks."))))
    return {"total":sum(p.numel() for p in model.parameters()),
        "trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "core":sum(p.numel() for p in core)}
