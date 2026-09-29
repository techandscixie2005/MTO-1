"""One-checkpoint eta0 plus the historically frozen affine calibration.

No fitting. f is the calibrated output. E and A are original auxiliary outputs;
calibrated f deliberately does not obey the original E/trace(A) identity.
"""
import sys
from pathlib import Path
import torch
from torch import nn

SOURCE = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
sys.path.insert(0, str(SOURCE / 'frozen_reference'))
from models_ea import MTOEA

ALPHA = 0.8511830211044088
BETA = 0.003725185373211049
HARTREE_EV = 27.211386245988


class CalibratedEta0(nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        self.base = MTOEA(config, normalization)
        self.register_buffer('alpha', torch.tensor(ALPHA, dtype=torch.float64))
        self.register_buffer('beta', torch.tensor(BETA, dtype=torch.float64))

    def calibrate(self, native_f):
        return torch.clamp_min(self.alpha * native_f.double() + self.beta, 0.0)

    def forward(self, **geometry):
        energy, matrix = self.base(**geometry)
        # Preserve operation order of the original FP64 metric export.
        native_f = (2 / 3) * (energy.double() / HARTREE_EV) * matrix.double().diagonal(
            dim1=-2, dim2=-1).sum(-1)
        return {'E': energy, 'A': matrix, 'f_native': native_f,
                'f': self.calibrate(native_f)}


def load_predictor(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['format'] == 'mto.calibrated_eta0.single_checkpoint.v1'
    model = CalibratedEta0(payload['config'], payload['normalization'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval()
