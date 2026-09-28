"""Pure pilot loss functions, no I/O or training entry point."""
import torch
C_F=2.0/(3.0*27.211386245988)
def terms(pred,target,stats,mode):
    energy,matrix=pred
    trace=matrix.diagonal(dim1=-2,dim2=-1).sum(-1)
    me=target['mask_E'].bool();ma=target['mask_A'].bool();valid=me&ma
    if not me.any() or not valid.any():raise ValueError('No valid E/trace labels')
    le=(energy[me]-target['E'][me]).square().mean()/stats['sE2']
    st=target['A'].diagonal(dim1=-2,dim2=-1).sum(-1)
    # Filter before arithmetic: masked NaNs never enter residuals/weights.
    residual=trace[valid]-st[valid]
    ls=residual.square().mean()/(3*stats['sA2'])
    w=target['E'][valid].detach().square()/stats['mean_E2_train']
    lw=(w*residual.square()).mean()/(3*stats['sA2'])
    if mode=='control':intensity=ls
    elif mode=='weighted':intensity=lw
    elif mode=='direct_f_matched':
        mf=valid&target['mask_f'].bool()
        if not mf.any():raise ValueError('No valid f labels')
        r=C_F*energy[mf]*trace[mf]-target['f'][mf]
        # Deliberately retain the gradient through predicted energy.
        d=C_F**2*3*stats['sA2']*stats['mean_E2_train']
        intensity=r.square().mean()/d
    else:raise ValueError('Unknown mode '+str(mode))
    return dict(total=le+intensity,energy=le,trace=ls,weighted_trace=lw,intensity=intensity)
