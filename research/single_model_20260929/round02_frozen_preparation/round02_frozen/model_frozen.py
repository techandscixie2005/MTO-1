"""Narrow cached-M path; all original parameters and buffers stay frozen."""
import hashlib
import math
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

PARENT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PARENT))
from architecture.model import AdapterMTOEA,SOURCE,TYPES,unpack,load_baseline,base_loss
from metrics import C_F

class FrozenAdapterMTO(AdapterMTOEA):
    def raw_states(self,z,pos,batch,n,edge_index=None):
        s,t=self.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        h=unpack(t,self.tensor_irreps);h['0e']=s[...,None]
        m,_=self.mto({key:h[key] for key in TYPES},z,batch,n,False)
        return m

    def from_raw(self,m):
        scalar,tensors=self.couple(m)
        d=self.decoder;v=d.trunk(scalar)
        energy=F.softplus(d.energy_head(v).squeeze(-1)+d.energy_offset)
        beta=d.beta_head(v).squeeze(-1)
        q=(d.tensor_gate(v).tanh()[...,None]*tensors).sum(-2)/math.sqrt(d.tensor_channels)
        Q=torch.einsum('...m,mij->...ij',q,d.cartesian_basis)
        C=beta[...,None,None]/math.sqrt(3)*d.identity+Q
        return energy,C@C.transpose(-1,-2)

    def forward(self,z,pos,batch,n,edge_index=None,return_aux=False):
        m=self.raw_states(z,pos,batch,n,edge_index)
        pred=self.from_raw(m)
        return (pred,m) if return_aux else pred

def build(cfg,stats,checkpoint_state,seed=11):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=FrozenAdapterMTO(cfg,stats,True)
    load_baseline(model,checkpoint_state)
    model.requires_grad_(False)
    model.right_adapter.requires_grad_(True)
    model.eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==4016
    return model

def tensor_digest(tensor):
    x=tensor.detach().cpu().contiguous()
    h=hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode())
    h.update(x.numpy().tobytes());return h.hexdigest()

def frozen_digests(model):
    # named_buffers includes nonpersistent element/mass constants.
    tensors={'parameter:'+k:p for k,p in model.named_parameters() if not k.startswith('right_adapter.')}
    tensors.update({'buffer:'+k:b for k,b in model.named_buffers()})
    return {k:tensor_digest(v) for k,v in tensors.items()}

def assert_optimizer(model,optimizer):
    expected={id(p) for p in model.right_adapter.parameters()}
    actual={id(p) for g in optimizer.param_groups for p in g['params']}
    assert expected==actual
    assert all(p.requires_grad==(id(p) in expected) for p in model.parameters())

def losses(pred,target,stats,variance,arm):
    result=base_loss(pred,target,stats)
    energy,matrix=pred
    f=C_F*energy*matrix.diagonal(dim1=-2,dim2=-1).sum(-1)
    valid=target['mask_f'].bool()
    assert bool(valid.any())
    lf=(f[valid]-target['f'][valid]).square().mean()/variance
    result['raw_f']=lf
    result['total']=result['energy']+(result['trace'] if arm=='trace' else lf)
    return result

@torch.no_grad()
def movement(model,m):
    raw={t:m[t][:,1:] for t in TYPES}
    transformed=model.right_adapter(raw)
    norm=torch.cat([raw[t].flatten(-2)/math.sqrt(d) for t,d in zip(TYPES,(1,3,5))],-1).norm(dim=-1)
    delta=torch.cat([(transformed[t]-raw[t]).flatten(-2)/math.sqrt(d) for t,d in zip(TYPES,(1,3,5))],-1).norm(dim=-1)
    ratio=delta/norm.clamp_min(1e-8)
    return {'mean_relative':float(ratio.mean()),'maximum_relative':float(ratio.max()),
            'rms_relative':float(ratio.square().mean().sqrt()),'mean_absolute':float(delta.mean())}
