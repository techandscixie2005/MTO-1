"""Full MTO-UV-T forward architecture, supervised ONLY by broadened spectra.

All quantities E, C, A and f are predictions. No electronic labels are inputs.
The baseline uses the same physical decoder after invariant molecular pooling.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from detanet_backbone import DetaNet

TYPES = ('0e', '1o', '2e')
EV_PER_HARTREE = 27.211386245988


def pool(x, batch, n):
    return x.new_zeros((n,) + x.shape[1:]).index_add(0, batch, x)


def invariants(m):
    return torch.cat([m['0e'].squeeze(-1), m['1o'].square().mean(-1),
                      m['2e'].square().mean(-1)], -1)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c = cfg['features']
        self.core = DetaNet(num_features=self.c, num_block=cfg['blocks'], maxl=2,
            num_radial=16, attention_head=4, rc=5., max_atomic_number=9,
            scalar_outsize=0, irreps_out=None, out_type='latent', summation=False,
            scale=None, device=torch.device('cpu'))
        for module, key in [(self.core.Embedding, 'elec'), (self.core, 'mass')]:
            value = getattr(module, key)
            delattr(module, key)
            module.register_buffer(key, value)

    def forward(self, z, pos, edge_index, batch):
        s, t = self.core(z=z, pos=pos, edge_index=edge_index, batch=batch)
        c = self.c
        h = {'0e': s[..., None], '1o': t[:, :3*c].reshape(-1,c,3),
             '2e': t[:, 3*c:8*c].reshape(-1,c,5)}
        # One invariant RMS per atom/type, shared across channels and all m.
        return {k: v / v.square().mean((-2,-1), keepdim=True).add(1e-8).sqrt()
                for k,v in h.items()}


class MolecularTensorOrbitals(nn.Module):
    def __init__(self, cfg, n_ref):
        super().__init__()
        self.channels, self.states = cfg['mto_channels'], cfg['states']
        c, f, q, w = self.channels, cfg['features'], cfg['query_dim'], cfg['router_hidden']
        self.gamma = 1 / math.sqrt(n_ref)
        self.project = nn.ModuleDict({t: nn.Linear(f,c,bias=False) for t in TYPES})
        self.query = nn.Embedding(self.states+1,q)
        self.router = nn.Sequential(nn.Linear(6*f+6+q,w), nn.SiLU(),
                                    nn.Linear(w,w), nn.SiLU(), nn.Linear(w,3*c))
        nn.init.normal_(self.router[-1].weight, std=.03)
        nn.init.normal_(self.router[-1].bias, std=.02)

    def forward(self, h, z, batch, n, export=False):
        u = invariants(h)
        count = pool(u.new_ones((len(z),1)), batch, n)
        elems = pool((z[:,None] == z.new_tensor([1,6,7,8,9])).to(u.dtype),batch,n)
        g = torch.cat([pool(u,batch,n)/count, count.log1p(), elems],-1)
        context = torch.cat([u,g[batch]],-1)
        inp = torch.cat([context[:,None].expand(-1,self.states+1,-1),
                         self.query.weight[None].expand(len(z),-1,-1)],-1)
        gates = self.router(inp).tanh().reshape(len(z),self.states+1,3,self.channels)
        m, terms, bases = {}, {}, {}
        for index,t in enumerate(TYPES):
            b = self.project[t](h[t].transpose(-1,-2)).transpose(-1,-2)
            f = gates[:,:,index,:,None]*b[:,None]
            m[t] = self.gamma*pool(f,batch,n)
            if export:
                terms[t], bases[t] = f, b
        return m, ({'mto':m,'assembly':terms,'bases':bases,'gates':gates} if export else {})


class ReferenceStateCoupling(nn.Module):
    """All seven parity/triangle-allowed paths into learned 0e and 2e channels."""
    def __init__(self, cfg):
        super().__init__()
        c = cfg['mto_channels']
        self.channels = c
        irreps = o3.Irreps(f'{c}x0e + {c}x1o + {c}x2e')
        self.tp = o3.FullyConnectedTensorProduct(irreps, irreps, f'{c}x0e + {c}x2e',
                                                irrep_normalization='component', path_normalization='element')
        self.scalar_dim, self.tensor_channels = 7*c, 2*c

    def forward(self, m):
        packed = torch.cat([m[t].flatten(-2) for t in TYPES],-1)
        p = self.tp(packed[:,:1].expand_as(packed[:,1:]), packed[:,1:])
        inv = invariants(m)
        scalar = torch.cat([inv[:,:1].expand_as(inv[:,1:]), inv[:,1:], p[...,:self.channels]],-1)
        tensors = torch.cat([p[...,self.channels:].reshape(*p.shape[:-1],self.channels,5), m['2e'][:,1:]],-2)
        return scalar, tensors, {'relation_0e':p[...,:self.channels],
                                 'relation_2e':tensors[...,:self.channels,:]}


class InvariantPooling(nn.Module):
    """B0: fixed sum pooling; invariant heads gate pooled 2e tensors per state."""
    def __init__(self, cfg, n_ref):
        super().__init__()
        self.gamma = 1/math.sqrt(n_ref)
        self.states = cfg['states']
        self.scalar_dim, self.tensor_channels = 3*cfg['features']+6, cfg['features']

    def forward(self, h, z, batch, n):
        m = {t:self.gamma*pool(h[t],batch,n) for t in TYPES}
        count = pool(h['0e'].new_ones((len(z),1)),batch,n)
        elems = pool((z[:,None] == z.new_tensor([1,6,7,8,9])).to(h['0e'].dtype),batch,n)
        scalar = torch.cat([invariants(m),count.log1p(),elems],-1)
        return scalar, m['2e'][:,None].expand(-1,self.states,-1,-1), {'pooled_tensors':m}


class PhysicalSpectrumDecoder(nn.Module):
    def __init__(self, cfg, n_in, tensor_channels, hidden, pooled=False, spectrum_rms=1.):
        super().__init__()
        self.states, self.pooled = cfg['states'], pooled
        self.tensor_channels = tensor_channels
        self.trunk = nn.Sequential(nn.LayerNorm(n_in),nn.Linear(n_in,hidden),nn.SiLU())
        repeats = self.states if pooled else 1
        self.energy_head = nn.Linear(hidden,repeats)
        self.beta_head = nn.Linear(hidden,repeats)
        self.tensor_gate = nn.Linear(hidden,repeats*tensor_channels)
        initial_e = torch.linspace(cfg['initial_energy_min_eV'],cfg['initial_energy_max_eV'],self.states)
        self.energy_offset = nn.Parameter(torch.log(torch.expm1(initial_e)))
        for head in (self.energy_head,self.beta_head,self.tensor_gate):
            nn.init.normal_(head.weight,std=.005)
        nn.init.zeros_(self.energy_head.bias)
        nn.init.constant_(self.beta_head.bias,.25)
        nn.init.normal_(self.tensor_gate.bias,std=.01)
        # Derive the Cartesian basis with e3nn, never hand-fill five components.
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            rtp = o3.ReducedTensorProducts('ij=ji', i='1o')
            assert str(rtp.irreps_out) == '1x0e+1x2e', str(rtp.irreps_out)
            basis = rtp.change_of_basis[1:].clone()
        finally:
            torch.set_default_dtype(old_dtype)
        self.register_buffer('cartesian_basis',basis.to(old_dtype))
        self.register_buffer('identity',torch.eye(3))
        self.register_buffer('energy_grid',torch.linspace(cfg['energy_min_eV'],cfg['energy_max_eV'],cfg['bins']))
        self.register_buffer('spectrum_rms',torch.tensor(float(spectrum_rms),dtype=torch.float64))
        self.sigma = cfg['sigma_eV']

    def broaden(self, energy, strength, grid=None):
        grid = self.energy_grid if grid is None else grid
        kernel = torch.exp(-.5*((grid-energy[...,None])/self.sigma).square())/(self.sigma*math.sqrt(2*math.pi))
        return (strength[...,None]*kernel).sum(-2), kernel

    def forward(self, scalar, tensors, export=False):
        hidden = self.trunk(scalar)
        energy_raw, beta, gate = self.energy_head(hidden), self.beta_head(hidden), self.tensor_gate(hidden)
        if not self.pooled:
            energy_raw, beta = energy_raw.squeeze(-1), beta.squeeze(-1)
        gate = gate.reshape(*tensors.shape[:-1]).tanh()
        energy = F.softplus(energy_raw+self.energy_offset)  # E_* = 1 eV
        q = (gate[...,None]*tensors).sum(-2)/math.sqrt(self.tensor_channels)
        q_cart = torch.einsum('...m,mij->...ij',q,self.cartesian_basis)
        factor = beta[...,None,None]/math.sqrt(3)*self.identity + q_cart
        strength_tensor = factor @ factor.transpose(-1,-2)
        trace = strength_tensor.diagonal(dim1=-2,dim2=-1).sum(-1)
        oscillator_strength = (2/3)*(energy/EV_PER_HARTREE)*trace
        source_spectrum,kernel = self.broaden(energy,oscillator_strength)
        spectrum = source_spectrum/self.spectrum_rms.to(source_spectrum.dtype)
        if not export:
            return spectrum, {}
        polarized = torch.einsum('bka,bkij->baij',kernel,2*(energy/EV_PER_HARTREE)[...,None,None]*strength_tensor)
        # Sort only exported state records; no non-differentiable assignment in loss.
        order = energy.argsort(-1)
        return spectrum, {'energy_eV':energy,'energy_order':order,'beta':beta,'Q_2e':q,
                          'Q_cartesian':q_cart,'C':factor,'A':strength_tensor,'trace_A':trace,
                          'f':oscillator_strength,'source_spectrum':source_spectrum,
                          'polarized_spectrum':polarized}


class SpectrumModel(nn.Module):
    def __init__(self, cfg, n_ref, variant='mto', baseline_hidden=None, spectrum_rms=1.):
        super().__init__()
        self.backbone, self.variant = Backbone(cfg), variant
        if variant == 'mto':
            self.readout = MolecularTensorOrbitals(cfg,n_ref)
            self.coupling = ReferenceStateCoupling(cfg)
            n_in,t_channels,hidden = self.coupling.scalar_dim,self.coupling.tensor_channels,cfg['head_hidden']
        elif variant == 'baseline':
            self.readout = InvariantPooling(cfg,n_ref)
            n_in,t_channels,hidden = self.readout.scalar_dim,self.readout.tensor_channels,baseline_hidden
        else:
            raise ValueError(variant)
        self.hidden = hidden
        self.decoder = PhysicalSpectrumDecoder(cfg,n_in,t_channels,hidden,variant=='baseline',spectrum_rms)

    def forward(self, z, pos, edge_index, batch, n, export=False):
        h = self.backbone(z,pos,edge_index,batch)
        if self.variant == 'mto':
            m,extra = self.readout(h,z,batch,n,export)
            scalar,tensors,relations = self.coupling(m)
            if export: extra.update(relations)
        else:
            scalar,tensors,extra = self.readout(h,z,batch,n)
        spectrum,physical = self.decoder(scalar,tensors,export)
        if export:
            extra.update(physical)
            return spectrum,extra
        return spectrum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_pair(cfg, n_ref, bins=601, seed=11, spectrum_rms=1.):
    assert bins == cfg['bins']
    torch.manual_seed(seed)
    mto = SpectrumModel(cfg,n_ref,spectrum_rms=spectrum_rms)
    target,shared = count_parameters(mto),count_parameters(mto.backbone)
    n_in,outputs = 3*cfg['features']+6, cfg['states']*(cfg['features']+2)
    def expected(w):
        return shared+2*n_in+(n_in+1)*w+(w+1)*outputs+cfg['states']
    width = min(range(8,4097), key=lambda w:abs(expected(w)-target))
    torch.manual_seed(seed)
    baseline = SpectrumModel(cfg,n_ref,'baseline',width,spectrum_rms)
    baseline.backbone.load_state_dict(mto.backbone.state_dict())
    counts = {'mto':target,'baseline':count_parameters(baseline),'backbone':shared,
              'mto_head_hidden':mto.hidden,'baseline_head_hidden':width}
    assert counts['baseline'] == expected(width)
    counts['relative_gap'] = abs(counts['baseline']-target)/target
    assert counts['relative_gap'] < .01, counts
    return {'baseline':baseline,'mto':mto},counts
