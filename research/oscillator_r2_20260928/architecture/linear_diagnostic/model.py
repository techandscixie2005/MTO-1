"""MTO readout screen; source model and old experiments stay unmodified."""
import json,math,pathlib,sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parent
SRC=pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
sys.path.insert(0,str(SRC))
from model_factory import build as build_source
from models import unpack,TYPES
C_F=2.0/(3.0*27.211386245988)
class ArchitectureMTO(nn.Module):
    def __init__(self,arm,stats,source_checkpoint=True):
        super().__init__()
        if arm not in ('original','direct_f','independent_trace'):raise ValueError(arm)
        self.arm=arm
        cfg=json.loads((SRC/'configs/mto_eta0.json').read_text())
        self.base=build_source(cfg,stats)
        if source_checkpoint:
            ck=torch.load(SRC/'runs/mto_eta0/best.pt',map_location='cpu',weights_only=False)
            assert ck['config']==cfg and ck['epoch']==33
            self.base.load_state_dict(ck['model'])
        if arm!='original':
            self.strength_head=nn.Linear(cfg['head_hidden'],1)
            nn.init.zeros_(self.strength_head.weight)
            scale=stats['f_std_train'] if arm=='direct_f' else stats['trace_std_train']
            mean=stats['f_mean_train'] if arm=='direct_f' else stats['trace_mean_train']
            self.register_buffer('strength_scale',torch.tensor(scale,dtype=torch.float32))
            with torch.no_grad():
                x=torch.tensor(max(mean/scale,1e-8))
                self.strength_head.bias.copy_((x+torch.log(-torch.expm1(-x))).reshape(1))
            if arm=='direct_f':
                # These discarded tensor-readout parameters are not used by direct f.
                del self.base.decoder.beta_head
                del self.base.decoder.tensor_gate
            else:
                # No shape loss in this screen; preserve the old shape readout as diagnostics.
                for head in (self.base.decoder.beta_head,self.base.decoder.tensor_gate):
                    for parameter in head.parameters():parameter.requires_grad_(False)
    def features(self,z,pos,batch,n,edge_index=None):
        b=self.base
        s,t=b.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        atom=unpack(t,b.tensor_irreps);atom['0e']=s[...,None]
        m,_=b.mto({k:atom[k] for k in TYPES},z,batch,n,False)
        scalar,tensors,_=b.cg(m)
        hidden=b.decoder.trunk(scalar)
        energy=F.softplus(b.decoder.energy_head(hidden).squeeze(-1)+b.decoder.energy_offset)
        return hidden,tensors,energy
    def original_matrix(self,hidden,tensors):
        d=self.base.decoder
        beta=d.beta_head(hidden).squeeze(-1)
        q=(d.tensor_gate(hidden).tanh()[...,None]*tensors).sum(-2)/math.sqrt(d.tensor_channels)
        Q=torch.einsum('...m,mij->...ij',q,d.cartesian_basis)
        C=beta[...,None,None]*d.identity/math.sqrt(3)+Q
        return C@C.transpose(-1,-2)
    def scalar_strength(self,hidden):
        return self.strength_scale*F.softplus(self.strength_head(hidden).squeeze(-1))
    def forward(self,z,pos,batch,n,edge_index=None):
        hidden,tensors,energy=self.features(z,pos,batch,n,edge_index)
        if self.arm=='original':
            A=self.original_matrix(hidden,tensors)
            trace=A.diagonal(dim1=-2,dim2=-1).sum(-1)
            f=C_F*energy*trace
            return dict(E=energy,f=f,A=A,trace=trace)
        strength=self.scalar_strength(hidden)
        if self.arm=='direct_f':
            return dict(E=energy,f=strength,A=None,trace=None)
        # Positive scalar magnitude, independent of normalized PSD tensor shape.
        # Shape export only: no supervision or gradient path from f through its normalization.
        raw=self.original_matrix(hidden,tensors)
        raw_trace=raw.diagonal(dim1=-2,dim2=-1).sum(-1)
        eps=1e-8
        shape=(raw+eps*self.base.decoder.identity/3)/(raw_trace[...,None,None]+eps)
        A=strength[...,None,None]*shape
        return dict(E=energy,f=C_F*energy*strength,A=A,trace=strength,shape=shape)
    def parameter_counts(self):
        return dict(total=sum(v.numel() for v in self.parameters()),trainable=sum(v.numel() for v in self.parameters() if v.requires_grad),backbone=sum(v.numel() for v in self.base.core.parameters()))
def build(arm,stats,load_initial=True):
    model=ArchitectureMTO(arm,stats)
    if load_initial:
        ck=torch.load(ROOT/'initial'/f'{arm}.pt',map_location='cpu',weights_only=False)
        assert ck['arm']==arm
        model.load_state_dict(ck['model'])
    return model
