"""Frozen original MTO or P-only polar-vector outer-product readout."""
import math,pathlib,sys
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen_reference'))
from models_ea import MTOEA
from models import make_core,MolecularTensorOrbitals,pack,unpack,TYPES

class PCoupling(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.irreps=o3.Irreps(f'{c}x0e + {c}x1o + {c}x2e')
        self.tp=o3.FullyConnectedTensorProduct(self.irreps,self.irreps,f'{c}x0e + {c}x1o',
            irrep_normalization='component',path_normalization='element')
    def forward(self,m):
        x=pack(m,self.irreps)
        # Product of pooled M0 and Mk retains every atomic pair, including i=j.
        return unpack(self.tp(x[:,:1].expand_as(x[:,1:]),x[:,1:]),self.tp.irreps_out)

class PDecoder(nn.Module):
    def __init__(self,stats,c):
        super().__init__()
        self.c=c  # plain attribute, not a buffer: must not enter state_dict
        self.trunk=nn.Sequential(nn.LayerNorm(c),nn.Linear(c,128),nn.SiLU())
        self.energy_head=nn.Linear(128,1)
        self.energy_offset=nn.Parameter(torch.log(torch.expm1(torch.tensor(stats['E_state_mean']))))
        self.vector_gate=nn.Linear(128,c)
        nn.init.normal_(self.energy_head.weight,std=.005);nn.init.zeros_(self.energy_head.bias)
        nn.init.normal_(self.vector_gate.weight,std=.005);nn.init.constant_(self.vector_gate.bias,.25)
        self.register_buffer('alpha_init',torch.tensor(1.))
    def forward(self,p):
        h=self.trunk(p['0e'].squeeze(-1))
        e=F.softplus(self.energy_head(h).squeeze(-1)+self.energy_offset)
        mu=self.alpha_init*(self.vector_gate(h).tanh()[...,None]*p['1o']).sum(-2)/math.sqrt(self.c)
        return e,mu[..., :,None]*mu[...,None,:],mu

class POuter(nn.Module):
    def __init__(self,cfg,stats):
        super().__init__();c=cfg['mto_channels']
        self.core=make_core(latent=True)
        self.tensor_irreps=self.core.blocks[-1].update.outt.irreps_out
        self.mto=MolecularTensorOrbitals(cfg,stats['n_ref'],False)
        self.cg=PCoupling(c);self.decoder=PDecoder(stats,c)
    def forward(self,z,pos,batch,n,edge_index=None,export=False):
        s,t=self.core(z=z,pos=pos,edge_index=edge_index,batch=batch)
        h=unpack(t,self.tensor_irreps);h['0e']=s[...,None]
        m,_=self.mto({k:h[k] for k in TYPES},z,batch,n,False)
        result=self.decoder(self.cg(m))
        return result if export else result[:2]

def build(cfg,stats,initial=True):
    torch.manual_seed(cfg['seed'])
    model=(POuter if cfg['architecture']=='p_outer' else MTOEA)(cfg,stats)
    if initial:
        model.load_state_dict(torch.load(ROOT/'initial'/('p_outer.pt' if cfg['architecture']=='p_outer' else 'original.pt'),map_location='cpu',weights_only=True))
    if not cfg['supervise_E']:
        for name,p in model.named_parameters():
            if name.startswith('decoder.energy_head.') or name=='decoder.energy_offset':p.requires_grad_(False)
    return model
