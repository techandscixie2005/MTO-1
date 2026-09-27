import torch
EV_PER_HARTREE=27.211386245988

def losses(pred,target,stats,eta):
    # eta=0 is strictly trace-only; E labels are never read in this branch.
    E,A=pred;ma=target['mask_A'];a=A[ma];at=target['A'][ma]
    s=a.diagonal(dim1=-2,dim2=-1).sum(-1);st=at.diagonal(dim1=-2,dim2=-1).sum(-1)
    ls=(s-st).square().mean()/(3*stats['sA2']) if s.numel() else A.sum()*0
    if eta==0:return ls,ls.detach()*0,ls,ls.detach()*0
    assert eta==1
    me=target['mask_E']
    le=(E[me]-target['E'][me]).square().mean()/stats['sE2'] if me.any() else E.sum()*0
    eye=torch.eye(3,device=A.device,dtype=A.dtype)
    q=a-s[:,None,None]*eye/3;qt=at-st[:,None,None]*eye/3
    lq=(q-qt).square().sum((-1,-2)).mean()/stats['sA2'] if s.numel() else A.sum()*0
    return le+ls+lq,le,ls,lq

def oscillator(pred):
    e,a=pred
    return (2/3)*(e/EV_PER_HARTREE)*a.diagonal(dim1=-2,dim2=-1).sum(-1)

def spectrum(e,f,mask):
    grid=torch.arange(1051,device=e.device,dtype=e.dtype)*.02
    e=torch.where(mask,e,torch.zeros_like(e));f=torch.where(mask,f,torch.zeros_like(f))
    return (f[...,None]*torch.exp(-.5*((grid-e[...,None])/.2).square())/(.2*(2*torch.pi)**.5)).sum(1)
