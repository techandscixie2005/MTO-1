"""Original DetaNet UV and planned MTO; all spectra are in source units.

The vendored DetaNet algorithm is unchanged. Only constant registration is adapted.
The global-gate diagnostic changes one local context selection, and no parameters.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from vendor.detanet_model.detanet import DetaNet

TYPES = ('0e', '1o', '2e')
VARIANTS = ('detanet_original_uv', 'detanet_mto_planned')
EV_PER_HARTREE = 27.211386245988
UV = dict(num_features=128, maxl=3, num_block=3, radial_type='trainable_bessel',
          num_radial=32, attention_head=8, rc=5., act='swish', dropout=0.,
          use_cutoff=False, max_atomic_number=9, atom_ref=None, scale=1.,
          scalar_outsize=240, irreps_out=None, summation=True, norm=False,
          out_type='scalar', grad_type=None)


def pool(x, batch, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add(0, batch, x)


def unpack(x, irreps):
    """Respect e3nn irrep-major, then channel-major, then m ordering."""
    parts = {}
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        v = x[..., sl].reshape(*x.shape[:-1], mul, ir.dim)
        key = str(ir)
        parts[key] = torch.cat((parts[key], v), -2) if key in parts else v
    return parts


def pack(parts, irreps):
    offsets, result = {}, []
    for mul, ir in irreps:
        key = str(ir)
        start = offsets.get(key, 0)
        result.append(parts[key][..., start:start+mul, :].flatten(-2))
        offsets[key] = start + mul
    return torch.cat(result, -1)


def invariants(h):
    return torch.cat((h['0e'].squeeze(-1), h['1o'].square().mean(-1),
                      h['2e'].square().mean(-1)), -1)


class CompatibleDetaNet(DetaNet):
    """Whitelist: nonpersistent elec/mass buffers; no changed forward operators."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for module, name in ((self.Embedding, 'elec'), (self, 'mass')):
            value = getattr(module, name)
            delattr(module, name)
            module.register_buffer(name, value, persistent=False)


def make_core(latent=False):
    cfg = dict(UV)
    if latent:
        cfg.update(scalar_outsize=0, out_type='latent', summation=False, scale=None)
    return CompatibleDetaNet(**cfg, device=torch.device('cpu'))


class MolecularTensorOrbitals(nn.Module):
    def __init__(self, cfg, n_ref, global_gate=False):
        super().__init__()
        self.global_gate = global_gate
        self.channels, self.states = cfg['mto_channels'], cfg['states']
        c, q, w = self.channels, cfg['query_dim'], cfg['router_hidden']
        f = UV['num_features']
        self.register_buffer('n_ref', torch.tensor(float(n_ref), dtype=torch.float64))
        self.project = nn.ModuleDict({t: nn.Linear(f, c, bias=False) for t in TYPES})
        self.query = nn.Embedding(self.states+1, q)
        self.router = nn.Sequential(nn.Linear(6*f+6+q, w), nn.SiLU(),
                                    nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 3*c))
        nn.init.normal_(self.router[-1].weight, std=.03)
        nn.init.normal_(self.router[-1].bias, std=.02)

    def forward(self, h, z, batch, n, export=False):
        u = invariants(h)
        count = pool(u.new_ones((len(z), 1)), batch, n)
        mean = pool(u, batch, n) / count
        elems = pool((z[:, None] == z.new_tensor([1, 6, 7, 8, 9])).to(u.dtype), batch, n)
        g = torch.cat((mean, count.log1p(), elems), -1)
        local = mean[batch] if self.global_gate else u
        context = torch.cat((local, g[batch]), -1)
        inp = torch.cat((context[:, None].expand(-1, self.states+1, -1),
                         self.query.weight[None].expand(len(z), -1, -1)), -1)
        gates = self.router(inp).tanh().reshape(len(z), self.states+1, 3, self.channels)
        m, terms, bases = {}, {}, {}
        gamma = self.n_ref.to(u.dtype).rsqrt()
        for j, t in enumerate(TYPES):
            b = self.project[t](h[t].transpose(-1, -2)).transpose(-1, -2)
            f = gates[:, :, j, :, None] * b[:, None]
            m[t] = gamma * pool(f, batch, n)
            if export:
                terms[t], bases[t] = f, b
        return m, dict(H=h, B=bases, F=terms, c=gates, M=m) if export else {}


class ReferenceStateCoupling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg['mto_channels']
        self.irreps = o3.Irreps([(c, t) for t in TYPES])
        self.tp = o3.FullyConnectedTensorProduct(self.irreps, self.irreps,
            [(c, '0e'), (c, '2e')], irrep_normalization='component', path_normalization='element')
        self.scalar_dim, self.tensor_channels = 7*c, 2*c

    def forward(self, m):
        x = pack(m, self.irreps)
        p = self.tp(x[:, :1].expand_as(x[:, 1:]), x[:, 1:])
        parts = unpack(p, self.tp.irreps_out)
        inv = invariants(m)
        scalar = torch.cat((inv[:, :1].expand_as(inv[:, 1:]), inv[:, 1:],
                            parts['0e'].squeeze(-1)), -1)
        tensors = torch.cat((parts['2e'], m['2e'][:, 1:]), -2)
        return scalar, tensors, parts


class PhysicalSpectrumDecoder(nn.Module):
    def __init__(self, cfg, n_in, tensor_channels, grid):
        super().__init__()
        self.tensor_channels = tensor_channels
        hidden = cfg['head_hidden']
        self.trunk = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.SiLU())
        self.energy_head, self.beta_head = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        self.tensor_gate = nn.Linear(hidden, tensor_channels)
        initial_e = torch.linspace(cfg['initial_energy_min_eV'], cfg['initial_energy_max_eV'], cfg['states'])
        self.energy_offset = nn.Parameter(torch.log(torch.expm1(initial_e)))
        for head in (self.energy_head, self.beta_head, self.tensor_gate):
            nn.init.normal_(head.weight, std=.005)
        nn.init.zeros_(self.energy_head.bias)
        nn.init.constant_(self.beta_head.bias, .25)
        nn.init.normal_(self.tensor_gate.bias, std=.01)
        # Build the basis in the actual construction dtype (double in precision tests).
        rtp = o3.ReducedTensorProducts('ij=ji', i='1o')
        slices = [sl for (_, ir), sl in zip(rtp.irreps_out, rtp.irreps_out.slices()) if str(ir) == '2e']
        assert len(slices) == 1
        self.register_buffer('cartesian_basis', rtp.change_of_basis[slices[0]].clone())
        self.register_buffer('identity', torch.eye(3))
        self.register_buffer('energy_grid', torch.as_tensor(grid, dtype=torch.get_default_dtype()))
        self.sigma = cfg['sigma_eV']

    def broaden(self, energy, strength, grid=None):
        grid = self.energy_grid if grid is None else grid
        kernel = torch.exp(-.5*((grid-energy[..., None])/self.sigma).square()) / (self.sigma*math.sqrt(2*math.pi))
        return (strength[..., None]*kernel).sum(-2)

    def forward(self, scalar, tensors, export=False):
        h = self.trunk(scalar)
        energy = F.softplus(self.energy_head(h).squeeze(-1)+self.energy_offset)  # E_* = 1 eV
        beta = self.beta_head(h).squeeze(-1)
        gates = self.tensor_gate(h).tanh()
        q = (gates[..., None]*tensors).sum(-2)/math.sqrt(self.tensor_channels)
        Q = torch.einsum('...m,mij->...ij', q, self.cartesian_basis)
        C = beta[..., None, None]/math.sqrt(3)*self.identity + Q
        A = C @ C.transpose(-1, -2)
        trace = A.diagonal(dim1=-2, dim2=-1).sum(-1)
        f = (2/3)*(energy/EV_PER_HARTREE)*trace
        spectrum = self.broaden(energy, f)
        return spectrum, dict(E=energy, beta=beta, Q_2e=q, Q=Q, C=C, A=A, f=f,
                              trace_A=trace, tensor_gates=gates) if export else {}


class SpectrumModel(nn.Module):
    def __init__(self, cfg, n_ref, grid, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.core = make_core(latent=variant != VARIANTS[0])
        # Metadata is taken from the actual output of the final block's equivariant map.
        self.tensor_irreps = self.core.blocks[-1].update.outt.irreps_out
        if variant != VARIANTS[0]:
            self.mto = MolecularTensorOrbitals(cfg, n_ref, False)
            self.cg = ReferenceStateCoupling(cfg)
            self.decoder = PhysicalSpectrumDecoder(cfg, self.cg.scalar_dim, self.cg.tensor_channels, grid)

    def forward(self, z, pos, batch, n, edge_index=None, export=False):
        out = self.core(z=z, pos=pos, edge_index=edge_index, batch=batch)
        if self.variant == VARIANTS[0]:
            return (out, {}) if export else out
        s, t = out
        h = unpack(t, self.tensor_irreps)
        h['0e'] = s[..., None]
        # 3o remains in every message/update block; only the MTO interface selects types.
        h = {key: h[key] for key in TYPES}
        m, extra = self.mto(h, z, batch, n, export)
        scalar, tensors, p = self.cg(m)
        spectrum, physical = self.decoder(scalar, tensors, export)
        if export:
            extra.update(CG=p, **physical)
            return spectrum, extra
        return spectrum


def build_pair(cfg, n_ref, grid, seed):
    torch.manual_seed(seed)
    a = SpectrumModel(cfg, n_ref, grid, VARIANTS[0])
    b = SpectrumModel(cfg, n_ref, grid, VARIANTS[1])
    common = {k: v for k, v in a.core.state_dict().items() if not k.startswith('sout.')}
    b.core.load_state_dict(common, strict=True)
    for key, value in b.core.state_dict().items():
        assert torch.equal(value, a.core.state_dict()[key]), key
    return dict(zip(VARIANTS, (a, b)))


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for k, p in model.core.named_parameters() if not k.startswith('sout.'))
    return dict(total=total, backbone=backbone, readout=total-backbone,
                active_observed=sum(p.numel() for p in model.parameters() if p.grad is not None))
