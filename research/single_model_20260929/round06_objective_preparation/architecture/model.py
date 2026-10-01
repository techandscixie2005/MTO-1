"""Shared right-operand adapter and raw-state decorrelation for frozen MTO.

The adapter is an equivariant learned readout map, not an identified physical
dipole operator. Decorrelation is a weak feature regularizer, not a claim that
the learned features are wavefunctions.
"""
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
sys.path.insert(0, str(SOURCE / 'frozen_reference'))
from models_ea import MTOEA
from models import TYPES, pack, unpack, invariants


class SharedRightAdapter(nn.Module):
    """F_t(x)=x+W_t[tanh(g_t(invariants(x))) * x].

    Channel matrices act only within equal irreps. Scalar gates are O(3)
    invariant. All parameters are shared across excited states. Zero W_t gives
    exactly the original function while W_t has a nonzero initial Jacobian.
    """
    def __init__(self, channels):
        super().__init__()
        c = channels
        self.channels = c
        self.gates = nn.Sequential(nn.LayerNorm(3*c), nn.Linear(3*c, 2*c),
                                   nn.SiLU(), nn.Linear(2*c, 3*c))
        self.mix = nn.ModuleDict({t: nn.Linear(c, c, bias=False) for t in TYPES})
        for layer in self.mix.values():
            nn.init.zeros_(layer.weight)

    def forward(self, right):
        inv = invariants(right)
        # Compress invariant dynamic range without applying a componentwise
        # nonlinear operation to any non-scalar irrep.
        inv = inv.sign() * inv.abs().log1p()
        gates = self.gates(inv).tanh().reshape(*inv.shape[:-1], 3, self.channels)
        return {t: right[t] + self.mix[t](
            (gates[..., j, :, None] * right[t]).transpose(-1, -2)
        ).transpose(-1, -2) for j, t in enumerate(TYPES)}


class AdapterMTOEA(MTOEA):
    def __init__(self, cfg, stats, adapter_enabled):
        super().__init__(cfg, stats)
        self.right_adapter = SharedRightAdapter(cfg['mto_channels'])
        self.adapter_enabled = bool(adapter_enabled)
        # Allocate identical checkpoint schemas in all arms. Dormant adapter
        # parameters are explicitly frozen in controls; this is not an active
        # capacity-matched control.
        self.right_adapter.requires_grad_(self.adapter_enabled)

    def couple(self, m):
        x = pack(m, self.cg.irreps)
        if self.adapter_enabled:
            right = self.right_adapter({t: m[t][:, 1:] for t in TYPES})
            xr = pack(right, self.cg.irreps)
        else:
            xr = x[:, 1:]
        parts = unpack(self.cg.tp(x[:, :1].expand_as(x[:, 1:]), xr),
                       self.cg.tp.irreps_out)
        inv = invariants(m)
        scalar = torch.cat((inv[:, :1].expand_as(inv[:, 1:]), inv[:, 1:],
                            parts['0e'].squeeze(-1)), -1)
        tensors = torch.cat((parts['2e'], m['2e'][:, 1:]), -2)
        return scalar, tensors

    def forward(self, z, pos, batch, n, edge_index=None, return_aux=False):
        s, t = self.core(z=z, pos=pos, edge_index=edge_index, batch=batch)
        h = unpack(t, self.tensor_irreps)
        h['0e'] = s[..., None]
        h = {key: h[key] for key in TYPES}
        m, _ = self.mto(h, z, batch, n, False)
        scalar, tensors = self.couple(m)
        d = self.decoder
        v = d.trunk(scalar)
        energy = F.softplus(d.energy_head(v).squeeze(-1) + d.energy_offset)
        beta = d.beta_head(v).squeeze(-1)
        q = (d.tensor_gate(v).tanh()[..., None] * tensors).sum(-2) / math.sqrt(d.tensor_channels)
        Q = torch.einsum('...m,mij->...ij', q, d.cartesian_basis)
        C = beta[..., None, None] / math.sqrt(3) * d.identity + Q
        pred = (energy, C @ C.transpose(-1, -2))
        return (pred, m) if return_aux else pred


def build_model(cfg, stats, adapter_enabled, adapter_seed=11):
    # Model source initialization and adapter initialization are reproducible;
    # all inherited parameters are subsequently replaced by the same checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(adapter_seed)
        return AdapterMTOEA(cfg, stats, adapter_enabled)


def load_baseline(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {f'right_adapter.{k}' for k in model.right_adapter.state_dict()}
    assert set(missing) == expected, (missing, expected)
    assert not unexpected, unexpected
    return model


def decorrelation(m, valid, eps=1e-8):
    """Mean squared normalized raw-feature overlap over valid excited pairs.

    M0 is excluded. Each irrep block is divided by sqrt(2l+1), so equal
    component variance does not overweight higher-l blocks merely by dimension.
    No batch centering, learned metric, transformed features, mu, or A occurs.
    Low-norm states remain included using an explicit epsilon denominator.
    Invalid states are removed before arithmetic, including masked NaNs.
    """
    valid = valid.bool()
    pieces = []
    for t, dim in zip(TYPES, (1, 3, 5)):
        x = m[t][:, 1:]
        x = torch.where(valid[..., None, None], x, torch.zeros_like(x))
        pieces.append(x.flatten(-2) / math.sqrt(dim))
    x = torch.cat(pieces, -1)
    norm = x.square().sum(-1, keepdim=True).clamp_min(eps**2).sqrt()
    unit = x / norm
    gram = unit @ unit.transpose(-1, -2)
    k = valid.shape[-1]
    pair = valid[..., :, None] & valid[..., None, :]
    pair = pair & torch.triu(torch.ones(k, k, device=x.device, dtype=torch.bool), diagonal=1)
    if not bool(pair.any()):
        return x.sum() * 0
    return gram[pair].square().mean()


def base_loss(pred, target, stats):
    energy, matrix = pred
    me = target['mask_E'].bool()
    valid = me & target['mask_A'].bool()
    if not bool(me.any()) or not bool(valid.any()):
        raise ValueError('No valid energy or trace labels')
    le = (energy[me] - target['E'][me]).square().mean() / stats['sE2']
    st = target['A'].diagonal(dim1=-2, dim2=-1).sum(-1)
    sp = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    ls = (sp[valid] - st[valid]).square().mean() / (3 * stats['sA2'])
    return {'total': le + ls, 'energy': le, 'trace': ls}


def training_loss(model, x, target, stats, orth_lambda):
    pred, raw_m = model(**x, return_aux=True)
    out = base_loss(pred, target, stats)
    valid = target['mask_E'].bool() & target['mask_A'].bool() & target['mask_f'].bool()
    orth = decorrelation(raw_m, valid)
    out['decorrelation'] = orth
    out['base'] = out['total']
    out['total'] = out['base'] + orth_lambda * orth
    return pred, out
