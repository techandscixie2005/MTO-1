"""Two fixed intensity objectives on the same original E/PSD prediction."""
import torch
from fresh_model import base_loss
EV_PER_HARTREE=27.211386245988
F_CONST=2/(3*EV_PER_HARTREE)
F_VARIANCE=0.0025089892829484074


def loss_components(pred,target,stats,objective):
    assert objective in ('trace','raw_f')
    energy,matrix=pred
    assert energy.dtype==matrix.dtype==torch.float32,'Both training objectives are FP32'
    assert stats['f_variance']==F_VARIANCE
    # Retain original control arithmetic and masks exactly, including extraction
    # of linear target/predicted traces before valid indexing.
    original=base_loss(pred,target,stats)
    mf=target['mask_E'].bool() & target['mask_f'].bool()
    if not bool(mf.any()):raise ValueError('No valid printed raw-f labels')
    # Selection precedes the nonlinear E*trace(A) product and squared error.
    e=energy[mf]
    strength=matrix[mf].diagonal(dim1=-2,dim2=-1).sum(-1)
    observed=target['f'][mf].to(dtype=torch.float32)
    prediction=e*strength*F_CONST
    lf=(prediction-observed).square().mean()/F_VARIANCE
    total=original['total'] if objective=='trace' else original['energy']+lf
    return {'total':total,'energy':original['energy'],'trace':original['trace'],
            'raw_f':lf,'trace_base':original['total']}
