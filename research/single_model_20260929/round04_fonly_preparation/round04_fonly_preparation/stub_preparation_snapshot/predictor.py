"""One geometry-only original MTO with four scalar coefficient buffers."""
import copy
import torch
from backbone import MTOEA, tensor_hash
from scalar_map import SCALE, KNOTS

C_F = 2.0 / (3.0 * 27.211386245988)
FORMAT = 'round04_single_native_model_fixed_hinge'


class ScalarSplineMTO(torch.nn.Module):
    def __init__(self, config, stats, theta):
        super().__init__()
        self.base = MTOEA(config, stats)
        coefficients = torch.as_tensor(theta, dtype=torch.float64).clone()
        if coefficients.shape != (4,) or not torch.isfinite(coefficients).all():
            raise ValueError('Four finite coefficients required')
        self.register_buffer('coefficients', coefficients)
        self.register_buffer('scale', torch.tensor(SCALE, dtype=torch.float64))
        self.register_buffer('knots', torch.tensor(KNOTS.copy(), dtype=torch.float64))

    def forward(self, **geometry):
        energy, matrix = self.base(**geometry)
        native = C_F * energy.double() * matrix.double().diagonal(dim1=-2, dim2=-1).sum(-1)
        f = native
        design = torch.stack((torch.ones_like(f), f / self.scale,
                              torch.clamp_min((f-self.knots[0])/self.scale, 0),
                              torch.clamp_min((f-self.knots[1])/self.scale, 0)), dim=-1)
        raw = design @ self.coefficients
        return {'E': energy, 'A': matrix, 'native_f': native, 'f': torch.clamp_min(raw, 0)}


def export_record(base, config, stats, theta, provenance):
    model = ScalarSplineMTO(config, stats, theta)
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.requires_grad_(False).eval()
    return {'format': FORMAT, 'source_config': copy.deepcopy(config), 'stats': copy.deepcopy(stats),
            'model': model.state_dict(), 'geometry_only_inference': True,
            'native_full_baseline_input': True, 'parameter_count': 4,
            'base_tensor_sha256': tensor_hash(model.base.state_dict()),
            'base_buffer_tensor_sha256': tensor_hash(dict(model.base.named_buffers())),
            'provenance': copy.deepcopy(provenance)}


def load_predictor(checkpoint, device='cpu'):
    record = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if record['format'] != FORMAT or record['geometry_only_inference'] is not True:
        raise ValueError('Wrong single-checkpoint format')
    if record['native_full_baseline_input'] is not True or record['parameter_count'] != 4:
        raise ValueError('Wrong input/function contract')
    model = ScalarSplineMTO(record['source_config'], record['stats'], record['model']['coefficients'])
    model.load_state_dict(record['model'], strict=True)
    if float(model.scale) != SCALE or not torch.equal(model.knots, torch.tensor(KNOTS, dtype=torch.float64)):
        raise ValueError('Scale or knot mismatch')
    if tensor_hash(model.base.state_dict()) != record['base_tensor_sha256']:
        raise ValueError('Base tensor mismatch')
    if tensor_hash(dict(model.base.named_buffers())) != record['base_buffer_tensor_sha256']:
        raise ValueError('Base nonpersistent buffer mismatch')
    return model.requires_grad_(False).eval().to(device)
