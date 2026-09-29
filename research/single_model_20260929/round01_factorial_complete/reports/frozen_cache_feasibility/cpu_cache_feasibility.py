"""Synthetic CPU feasibility only; no dataset, fitting run or checkpoint output."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent
sys.path.insert(0, str(CAMPAIGN))
from architecture.model import SOURCE, TYPES, build_model, load_baseline, base_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_raw_m(model, raw):
    # Exact current post-M operators, for this test only. A future implementation
    # must share one helper between full and cached forward, not maintain copies.
    scalar, tensors = model.couple(raw)
    d = model.decoder
    v = d.trunk(scalar)
    energy = F.softplus(d.energy_head(v).squeeze(-1) + d.energy_offset)
    beta = d.beta_head(v).squeeze(-1)
    q = (d.tensor_gate(v).tanh()[..., None] * tensors).sum(-2) / math.sqrt(d.tensor_channels)
    Q = torch.einsum('...m,mij->...ij', q, d.cartesian_basis)
    C = beta[..., None, None] / math.sqrt(3) * d.identity + Q
    return energy, C @ C.transpose(-1, -2)


def maximum_difference(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260930)
    checkpoint = SOURCE / 'runs/mto_eta0/best.pt'
    assert sha(checkpoint) == '9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136'
    cfg = json.loads((SOURCE / 'configs/mto_eta0.json').read_text())
    stats = json.loads((SOURCE / 'data/normalization.json').read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = load_baseline(build_model(cfg, stats, True, 11), state['model']).eval()
    del state
    model.requires_grad_(False)
    model.right_adapter.requires_grad_(True)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if not name.startswith('right_adapter.')}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    params = list(model.right_adapter.parameters())
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(name.startswith('right_adapter.') for name in names)
    dropout = {name: module.p for name, module in model.named_modules() if isinstance(module, nn.Dropout)}
    batchnorm = [name for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    assert all(p == 0 for p in dropout.values()) and not batchnorm
    # Synthetic water and a separate H2; all directed intramolecule edges.
    x = {'z': torch.tensor([8, 1, 1, 1, 1]),
         'pos': torch.tensor([[0.,0.,0.],[.96,0.,0.],[-.24,.93,0.],[3.,0.,0.],[3.75,0.,0.]]),
         'batch': torch.tensor([0,0,0,1,1]), 'n': 2,
         'edge_index': torch.tensor([[0,0,1,1,2,2,3,4],[1,2,0,2,0,1,4,3]])}
    with torch.no_grad():
        initial, raw = model(**x, return_aux=True)
    cache = {key: value.detach().clone() for key, value in raw.items()}
    assert all(not value.requires_grad for value in cache.values())
    initial_cached = decode_raw_m(model, cache)
    assert maximum_difference(initial, initial_cached) == 0
    # Test a nonidentity adapter, not only the trivial initial identity.
    with torch.no_grad():
        for layer in model.right_adapter.mix.values():
            layer.weight.normal_(std=.02)
    full = model(**x)
    cached = decode_raw_m(model, cache)
    forward_error = maximum_difference(full, cached)
    assert forward_error == 0
    target = {'E': initial[0].detach() * .97, 'A': initial[1].detach() * 1.03,
              'mask_E': torch.ones_like(initial[0], dtype=torch.bool),
              'mask_A': torch.ones_like(initial[0], dtype=torch.bool)}
    loss_full = base_loss(full, target, stats)['total']
    loss_cached = base_loss(cached, target, stats)['total']
    grad_full = torch.autograd.grad(loss_full, params)
    grad_cached = torch.autograd.grad(loss_cached, params)
    gradient_error = maximum_difference(grad_full, grad_cached)
    assert gradient_error == 0 and float(loss_full - loss_cached) == 0
    energy, matrix = decode_raw_m(model, cache)
    gradient_e = torch.autograd.grad(energy.sum(), params, retain_graph=True)
    gradient_a = torch.autograd.grad(matrix.diagonal(dim1=-2, dim2=-1).sum(), params)
    norm_e = float(torch.sqrt(sum(g.square().sum() for g in gradient_e)))
    norm_a = float(torch.sqrt(sum(g.square().sum() for g in gradient_a)))
    assert norm_e > 0 and norm_a > 0
    optimizer = torch.optim.Adam(params, lr=1e-5, amsgrad=True, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    synthetic_loss = base_loss(decode_raw_m(model, cache), target, stats)['total']
    synthetic_loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
    optimizer.step()  # One synthetic test step, discarded; no fit or saved weights.
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in params}
    assert all(torch.equal(p, frozen[name]) and p.grad is None
               for name, p in model.named_parameters() if name in frozen)
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    post_step_error = maximum_difference(model(**x), decode_raw_m(model, cache))
    assert post_step_error == 0
    values_per_molecule = sum(value[0].numel() for value in cache.values())
    report = {'passed': True, 'synthetic_molecules': 2, 'dataset_loaded': False,
              'qc_labels_loaded': False, 'test_evaluated': False, 'gpu_used': False,
              'fit_launched': False, 'checkpoint_written': False,
              'synthetic_optimizer_steps_discarded': 1,
              'initial_identity_output_max_abs_error': 0.0,
              'nonidentity_output_max_abs_error': forward_error,
              'LE_Ls_adapter_gradient_max_abs_error': gradient_error,
              'post_synthetic_step_output_max_abs_error': post_step_error,
              'energy_sum_adapter_gradient_norm': norm_e,
              'trace_A_sum_adapter_gradient_norm': norm_a,
              'frozen_parameters_bitwise_unchanged': True,
              'all_named_buffers_including_nonpersistent_bitwise_unchanged': True,
              'optimizer_only_adapter_parameters': True,
              'adapter_trainable_parameters': sum(p.numel() for p in params),
              'dropout_probabilities': dropout, 'batchnorm_modules': batchnorm,
              'cache_shapes': {key: list(value.shape) for key, value in cache.items()},
              'float32_bytes_per_molecule': values_per_molecule * 4,
              'estimated_120355_training_molecule_cache_bytes': values_per_molecule * 4 * 120355,
              'source_hashes': {str(path): sha(path) for path in
                  (Path(__file__), CAMPAIGN/'architecture/model.py', SOURCE/'frozen_reference/models_ea.py',
                   SOURCE/'frozen_reference/upstream/models.py', SOURCE/'configs/mto_eta0.json',
                   SOURCE/'data/normalization.json')},
              'limitations': ['Synthetic CPU parity only; no real TRAIN cache or GPU parity checked',
                  'Same geometry and edge/batch arithmetic; cross-device/cache-batch numerical tolerance still needed',
                  'No architecture or optimization schedule selected']}
    (ROOT / 'CPU_CACHE_FEASIBILITY.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('passed','nonidentity_output_max_abs_error',
          'LE_Ls_adapter_gradient_max_abs_error','post_synthetic_step_output_max_abs_error',
          'energy_sum_adapter_gradient_norm','trace_A_sum_adapter_gradient_norm',
          'frozen_parameters_bitwise_unchanged','all_named_buffers_including_nonpersistent_bitwise_unchanged',
          'float32_bytes_per_molecule','estimated_120355_training_molecule_cache_bytes')}))


if __name__ == '__main__':
    main()
