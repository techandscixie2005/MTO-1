"""Shared phase-blind PSD transform. Tensor directions are latent features."""
import math
import torch
from torch import nn

MODES = ('original', 'scalar', 'tensor')
ELEMENTS = (1, 6, 7, 8, 9)
ENERGY_EV_PER_HARTREE = 27.211386245988
EPSILON = 1e-6
STRENGTH_SCALE = 1.0
BOUND = .25
CONTRACT = {
    'version': 'round07_shared_psd_congruence_v1', 'states': 10,
    'energy_eV_per_Hartree': ENERGY_EV_PER_HARTREE, 'epsilon': EPSILON,
    'strength_scale': STRENGTH_SCALE, 'bound': BOUND,
    'gate': 'Linear(9,16)-SiLU-Linear(16,1)', 'gate_parameters': 177,
    'features': ['log1p(N_H)', 'log1p(N_C)', 'log1p(N_N)', 'log1p(N_O)',
                 'log1p(N_F)', 'mean(E_H)', 'log1p(S/s0)', 'log1p(sum(E_H*s)/s0)', 'q'],
    'B': '(sum(A)+epsilon*I/3)/(sum(trace(A))+epsilon)',
    'q': 'Frobenius(B)/sqrt(3)', 'all_ten_prediction_slots': True,
    'features_detached': False, 'labels_or_masks_in_features': False,
    'scalar': '(1+b*q)^2*A', 'tensor': '(I+b*B)*A*(I+b*B)^T',
    'energy_forward_unchanged': True,
}


def element_counts(z, batch, n, dtype):
    assert z.ndim == batch.ndim == 1 and z.shape == batch.shape
    assert z.dtype == batch.dtype == torch.long
    assert bool(((batch >= 0) & (batch < n)).all())
    assert bool(torch.stack([z == element for element in ELEMENTS]).any(0).all())
    return torch.stack([torch.bincount(batch[z == element], minlength=n)
                        for element in ELEMENTS], -1).to(dtype=dtype)


class SharedPSDCongruence(nn.Module):
    """One shared 177-parameter scalar gate; no labels or state masks enter."""
    def __init__(self, mode):
        super().__init__()
        assert mode in MODES
        self.mode = mode
        self.gate = nn.Sequential(nn.Linear(9, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.gate.requires_grad_(mode != 'original')

    def context(self, energy, matrix, counts):
        assert energy.ndim == 2 and energy.shape[1] == 10
        assert matrix.shape == (*energy.shape, 3, 3)
        assert counts.shape == (len(energy), 5)
        assert energy.dtype == matrix.dtype == counts.dtype
        assert energy.device == matrix.device == counts.device
        assert bool(torch.isfinite(energy).all() & torch.isfinite(matrix).all()
                    & torch.isfinite(counts).all())
        strength = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
        assert bool((energy >= 0).all() & (strength >= 0).all() & (counts >= 0).all())
        aggregate = matrix.sum(1)
        total = aggregate.diagonal(dim1=-2, dim2=-1).sum(-1)
        identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        B = (aggregate + EPSILON * identity / 3) / (total + EPSILON)[:, None, None]
        q = B.square().sum((-2, -1)).div(3).sqrt()
        hartree = energy / ENERGY_EV_PER_HARTREE
        weighted = (hartree * strength / STRENGTH_SCALE).sum(1)
        features = torch.cat((counts.log1p(), hartree.mean(1)[:, None],
                              (total / STRENGTH_SCALE).log1p()[:, None],
                              weighted.log1p()[:, None], q[:, None]), -1)
        assert bool(torch.isfinite(features).all() & torch.isfinite(B).all())
        b = BOUND * self.gate(features).squeeze(-1).tanh()
        assert bool(torch.isfinite(b).all())
        return {'features': features, 'B': B, 'q': q, 'b': b,
                'untransformed_A': matrix, 'strength': strength}

    def forward(self, energy, matrix, counts):
        if self.mode == 'original':
            # The reference has no added objective graph. Its diagnostics are
            # computed consistently without influencing base gradients.
            with torch.no_grad():
                context = self.context(energy, matrix, counts)
            return matrix, context
        context = self.context(energy, matrix, counts)
        b, B, q = (context[key] for key in ('b', 'B', 'q'))
        if self.mode == 'scalar':
            transformed = (1 + b * q).square()[:, None, None, None] * matrix
        else:
            identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
            L = identity + b[:, None, None] * B
            transformed = L[:, None] @ matrix @ L.transpose(-1, -2)[:, None]
        assert bool(torch.isfinite(transformed).all())
        return transformed, context
