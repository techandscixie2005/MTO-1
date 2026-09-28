"""Common loss for the architecture-only screen."""
import torch
C_F=2.0/(3.0*27.211386245988)
def terms(pred,target,stats):
    me=target['mask_E'].bool();mf=target['mask_f'].bool()
    if not me.any() or not mf.any():raise ValueError('No valid E/f labels')
    le=(pred['E'][me]-target['E'][me]).square().mean()/stats['sE2']
    lf=(pred['f'][mf]-target['f'][mf]).square().mean()/stats['f_var_train']
    return dict(total=le+lf,energy=le,intensity=lf)
def native_f64(pred,arm):
    # Original must sum tensor diagonals after casting, matching historical exporter.
    e=pred['E'].double()
    if arm=='original':return C_F*e*pred['A'].double().diagonal(dim1=-2,dim2=-1).sum(-1)
    if arm=='retained_residual':
        base=C_F*e*pred['base_A'].double().diagonal(dim1=-2,dim2=-1).sum(-1)
        return (base+pred['delta_f'].double()).abs()
    if arm=='independent_trace':return C_F*e*pred['trace'].double()
    return pred['f'].double()
