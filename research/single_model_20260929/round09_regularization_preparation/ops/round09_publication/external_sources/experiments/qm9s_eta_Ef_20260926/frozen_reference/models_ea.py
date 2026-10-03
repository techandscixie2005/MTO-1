"""Pinned full DetaNet+MTO and a full DetaNet with native atomwise E/A heads."""
import math,pathlib,sys
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
sys.path.insert(0,str(pathlib.Path(__file__).parent/'upstream'))
from models import (SpectrumModel,make_core,unpack,pool,TYPES)
from vendor.detanet_model.modules import MLP,Equivariant_Multilayer

class MTOEA(SpectrumModel):
    def __init__(self,cfg,stats):
        super().__init__(cfg,stats['n_ref'],[0.,21.],'detanet_mto_planned')
        with torch.no_grad():
            self.decoder.energy_offset.copy_(torch.log(torch.expm1(torch.tensor(stats['E_state_mean']))))
    def forward(self,z,pos,batch,n,edge_index=None):
        s,t=self.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        h=unpack(t,self.tensor_irreps);h['0e']=s[...,None];h={k:h[k] for k in TYPES}
        m,_=self.mto(h,z,batch,n,False)
        scalar,tensors,_=self.cg(m)
        d=self.decoder;v=d.trunk(scalar)
        E=F.softplus(d.energy_head(v).squeeze(-1)+d.energy_offset)
        beta=d.beta_head(v).squeeze(-1)
        q=(d.tensor_gate(v).tanh()[...,None]*tensors).sum(-2)/math.sqrt(d.tensor_channels)
        Q=torch.einsum('...m,mij->...ij',q,d.cartesian_basis)
        C=beta[...,None,None]/math.sqrt(3)*d.identity+Q
        return E,C@C.transpose(-1,-2)

class DetaNetEA(nn.Module):
    """Complete original backbone + native invariant/equivariant atomwise readout.

    No MTO query/router, orbital pooling, or reference-target CG coupling.
    Native DetaNet MLP/Equivariant_Multilayer generate 10 energies and 10 C tensors;
    molecular summation is explicit because original tuple-output forward cannot scatter tuples.
    Both models use the same softplus E and C C^T PSD output convention.
    """
    def __init__(self,cfg,stats):
        super().__init__();self.core=make_core(latent=True)
        irreps=self.core.blocks[-1].update.outt.irreps_out
        self.sout=MLP((128,128,20),act='swish',dropout=0.)
        self.tout=Equivariant_Multilayer([irreps,o3.Irreps('128x2e'),o3.Irreps('10x2e')],act='swish')
        self.energy_offset=nn.Parameter(torch.log(torch.expm1(torch.tensor(stats['E_state_mean']))))
        self.beta_offset=nn.Parameter(torch.full((10,),.25))
        self.register_buffer('gamma',torch.tensor(stats['n_ref']**-.5))
        rtp=o3.ReducedTensorProducts('ij=ji',i='1o')
        sl=[sl for (_,ir),sl in zip(rtp.irreps_out,rtp.irreps_out.slices()) if str(ir)=='2e'][0]
        self.register_buffer('basis',rtp.change_of_basis[sl].clone());self.register_buffer('identity',torch.eye(3))
        # Small initial property deviations, matching the MTO decoder's initial scale.
        last=[m for m in self.sout.modules() if isinstance(m,nn.Linear)][-1]
        nn.init.normal_(last.weight,std=.005);nn.init.zeros_(last.bias)
        with torch.no_grad():self.tout.e_mlp[-1].weight.mul_(.01)
    def forward(self,z,pos,batch,n,edge_index=None):
        s,t=self.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        scalars=pool(self.sout(s),batch,n)*self.gamma
        q=pool(self.tout(t),batch,n).reshape(n,10,5)*self.gamma
        E=F.softplus(scalars[:,:10]+self.energy_offset)
        C=(scalars[:,10:]+self.beta_offset)[...,None,None]/math.sqrt(3)*self.identity+torch.einsum('...m,mij->...ij',q,self.basis)
        return E,C@C.transpose(-1,-2)

def build(cfg,stats):
    torch.manual_seed(cfg['seed'])
    return (DetaNetEA if cfg['model']=='detanet' else MTOEA)(cfg,stats)

def losses(pred,target,stats):
    E,A=pred;Et,At=target
    le=(E-Et).square().mean()/stats['sE2']
    la=(A-At).square().sum((-2,-1)).mean()/stats['sA2']
    return le+la,le,la
