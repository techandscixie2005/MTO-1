import torch
EV_PER_HARTREE=27.211386245988
def losses(pred,target,stats,eta):
    E,A=pred;me=target['mask_E'];ma=target['mask_A']
    # Select before subtraction: masked NaNs must not pollute values or gradients.
    le=(E[me]-target['E'][me]).square().mean()/stats['sE2']
    a=A[ma];at=target['A'][ma]
    s=a.diagonal(dim1=-2,dim2=-1).sum(-1);st=at.diagonal(dim1=-2,dim2=-1).sum(-1)
    eye=torch.eye(3,device=a.device,dtype=a.dtype)
    q=a-s[:,None,None]*eye/3;qt=at-st[:,None,None]*eye/3
    ls=(s-st).square().mean()/(3*stats['sA2'])
    lq=(q-qt).square().sum((-1,-2)).mean()/stats['sA2']
    return le+ls+eta*lq,le,ls,lq
def oscillator(pred):
    e,a=pred
    return (2/3)*(e/EV_PER_HARTREE)*a.diagonal(dim1=-2,dim2=-1).sum(-1)
def spectrum(e,f,mask):
    grid=torch.arange(1051,device=e.device,dtype=e.dtype)*.02
    e=torch.where(mask,e,torch.zeros_like(e));f=torch.where(mask,f,torch.zeros_like(f))
    return (f[...,None]*torch.exp(-.5*((grid-e[...,None])/.2).square())/(.2*(2*torch.pi)**.5)).sum(1)
