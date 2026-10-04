"""Scalar-gated same-irrep tensor transport; no electronic-state interpretation."""
import torch
from torch import nn
from torch_scatter import scatter

MODES=('original','local','neighbor')
CONTRACT={'version':'round08_transport_v1','modes':list(MODES),'blocks_one_based':[2,3],
    'irreps':['128x1o','128x2e','128x3o'],'channels':128,'theta_shape_per_block':[3,128],
    'added_parameters':768,'theta_initialization':'zeros','coefficient':'tanh(theta)',
    'edge_gate':'tanh(existing mijs2)','source_index':0,'receiver_index':1,
    'input':'pre-block T','normalization':'incoming edge count clamped minimum1',
    'insertion':'after original outt(scatter(mijt)), before local uattn',
    'first_block':'unchanged','empty_neighborhood':'zero added residual',
    'original':'frozen theta and bypass','active_zero_theta_shortcut':False,
    'physical_phase_or_density_claim':False,'diagnostic_norm_denominator_floor':1e-12}


class TensorTransport(nn.Module):
    def __init__(self,irreps,mode):
        super().__init__();assert mode in MODES
        assert [(int(m),str(ir)) for m,ir in irreps]==[(128,'1o'),(128,'2e'),(128,'3o')]
        self.mode=mode;self.slices=tuple(irreps.slices());self.dimensions=(3,5,7)
        self.theta=nn.Parameter(torch.zeros(3,128),requires_grad=mode!='original')

    def forward(self,T,edge_scalar,index):
        assert T.ndim==2 and T.shape[1]==1920
        assert edge_scalar.shape==(index.shape[1],128) and index.shape[0]==2
        if self.mode=='original':return torch.zeros_like(T)
        source,receiver=index;chosen=source if self.mode=='neighbor' else receiver
        degree=T.new_zeros(len(T)).index_add(0,receiver,T.new_ones(len(receiver))).clamp_min(1)
        gate=edge_scalar.tanh();parts=[]
        for l,(sl,dim) in enumerate(zip(self.slices,self.dimensions)):
            x=T[:,sl].reshape(len(T),128,dim)
            weighted=gate[...,None]*x[chosen]
            mean=x.new_zeros(x.shape).index_add(0,receiver,weighted)/degree[:,None,None]
            parts.append((self.theta[l].tanh()[None,:,None]*mean).flatten(1))
        return torch.cat(parts,-1)


class TransportBlock(nn.Module):
    """Retains existing message/update names, tensors and one Attention call."""
    def __init__(self,old,mode):
        super().__init__();self.message=old.message;self.update=old.update
        self.transport=TensorTransport(self.update.outt.irreps_out,mode)
        self.record_diagnostics=False;self.last_diagnostic={}

    def forward(self,S,T,rbf,sh,index):
        # This is the existing Message arithmetic, exposing its actual mijs2.
        edge=self.message.Attention(S=S,rbf=rbf,index=index)
        mijs2,mijs=torch.split(edge,[self.message.feature,self.message.feature],dim=-1)
        mijt=self.message.tp(mijs2,sh)
        update=self.update;receiver=index[1]
        ut=update.outt(scatter(src=mijt,index=receiver,dim=0))
        us=update.actu(update.outs(scatter(src=mijs,index=receiver,dim=0)))
        delta=self.transport(T,mijs2,index)  # pre-block T, never T+ut
        # Original bypass keeps the original arithmetic, not T+ut+0.
        T_mid=T+ut if self.transport.mode=='original' else T+ut+delta
        S_mid=S+update.drop(us)
        ut2,us2=update.uattn(T=T_mid,S=S_mid)
        self.last_diagnostic={}
        if self.record_diagnostics:
            with torch.no_grad():
                for l,(sl,dim) in enumerate(zip(self.transport.slices,(3,5,7)),1):
                    dn=delta[:,sl].norm(dim=-1);un=ut[:,sl].norm(dim=-1);tn=T[:,sl].norm(dim=-1)
                    # Explicit sums/counts give atom-weighted epoch summaries.
                    for label,values in [('delta_norm',dn),('original_message_norm',un),
                                         ('preblock_T_norm',tn),
                                         ('regularized_relative_to_message',dn/un.clamp_min(1e-12)),
                                         ('regularized_relative_to_preblock',dn/tn.clamp_min(1e-12))]:
                        prefix=f'l{l}_{label}'
                        self.last_diagnostic[prefix+'_sum']=float(values.sum())
                        self.last_diagnostic[prefix+'_count']=values.numel()
                        self.last_diagnostic[prefix+'_min']=float(values.min())
                        self.last_diagnostic[prefix+'_max']=float(values.max())
                    self.last_diagnostic[f'l{l}_message_zero_count']=int((un==0).sum())
                    self.last_diagnostic[f'l{l}_preblock_zero_count']=int((tn==0).sum())
                    self.last_diagnostic[f'l{l}_message_below_floor_count']=int((un<1e-12).sum())
                    self.last_diagnostic[f'l{l}_preblock_below_floor_count']=int((tn<1e-12).sum())
                coefficient=self.transport.theta.tanh()
                self.last_diagnostic.update(coefficient_min=float(coefficient.min()),
                    coefficient_max=float(coefficient.max()),theta_l2_mean=float(self.transport.theta.norm()))
        return S_mid+update.drop(us2),T_mid+ut2
