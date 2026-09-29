"""CPU-only synthetic response algebra; no MTO data, checkpoint, or GPU access."""
import json
import math
import os
import platform
import numpy as np
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU isolation must be explicit'
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(20260930)
torch.manual_seed(20260930)
C = 2 / 3  # Synthetic excitation energies are already in Hartree.
report = {'seed': 20260930, 'device': 'cpu', 'torch': torch.__version__,
          'numpy': np.__version__, 'python': platform.python_version(),
          'actual_molecular_data_loaded': False, 'GPU_API_called': False}


def response(h, q):
    e, u = np.linalg.eigh(h)
    strength = np.diag(u.T @ q @ u)
    return e, strength, C * e * strength, u


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def maxerr(x, y):
    return float(np.max(np.abs(np.asarray(x) - np.asarray(y))))


# Shared-basis transformations and eigenvector sign gauge, away from degeneracy.
a = rng.normal(size=(4, 4))
h = a @ a.T + np.eye(4)
b = rng.normal(size=(4, 6))
q = b @ b.T
e, s, f, u = response(h, q)
v, _ = np.linalg.qr(rng.normal(size=(4, 4)))
er, sr, fr, _ = response(v.T @ h @ v, v.T @ q @ v)
us = u @ np.diag([1, -1, -1, 1])
permutation = np.eye(4)[[2, 0, 3, 1]]
_, sp, fp, _ = response(permutation @ h @ permutation.T, permutation @ q @ permutation.T)
report['full_response_invariance'] = {
    'common_basis_f_max_error': maxerr(fr, f),
    'eigenvector_sign_strength_max_error': maxerr(np.diag(us.T @ q @ us), s),
    'latent_permutation_f_max_error': maxerr(fp, f),
    'strength_sum_minus_trace_Q': abs(float(s.sum() - np.trace(q))),
    'f_sum_minus_two_thirds_trace_HQ': abs(float(f.sum() - C*np.trace(h @ q)))}
assert max(report['full_response_invariance'].values()) < 1e-11

# Eigh's mandatory ascending ordering breaks identity for an unsorted base head.
original_e = np.array([2., 1., 3.])
original_s = np.array([.2, .9, .5])
ordered_e, ordered_s, ordered_f, _ = response(np.diag(original_e), np.diag(original_s))
report['unsorted_identity_counterexample'] = {
    'original_E': original_e.tolist(), 'original_f': (C*original_e*original_s).tolist(),
    'eigh_E': ordered_e.tolist(), 'eigh_f': ordered_f.tolist(),
    'max_f_change_at_zero_adapter': maxerr(ordered_f, C*original_e*original_s)}
assert report['unsorted_identity_counterexample']['max_f_change_at_zero_adapter'] > .1

# At exact degeneracy different valid bases redistribute individual strengths.
hdeg = 2*np.eye(2)
qdeg = np.array([[1., .8], [.8, 1.]])
rot = rotation(math.pi/4)
s_identity = np.diag(qdeg)
s_rotated = np.diag(rot.T @ qdeg @ rot)
report['exact_degeneracy'] = {
    'same_H_residual': maxerr(rot.T @ hdeg @ rot, hdeg),
    'strengths_basis_I': s_identity.tolist(), 'strengths_basis_rotated': s_rotated.tolist(),
    'subspace_total_error': abs(float(s_identity.sum()-s_rotated.sum())),
    'same_total_f': float(C*2*s_identity.sum())}
assert maxerr(s_identity, s_rotated) > .7
assert report['exact_degeneracy']['subspace_total_error'] < 1e-14


def eigengrad(gap, theta_value=0.):
    theta = torch.tensor(theta_value, requires_grad=True)
    hh = torch.stack((torch.stack((theta.new_tensor(2-gap/2), theta)),
                      torch.stack((theta, theta.new_tensor(2+gap/2)))))
    _, uu = torch.linalg.eigh(hh)
    ss = torch.diagonal(uu.T @ torch.tensor(qdeg) @ uu)
    grad, = torch.autograd.grad(ss[0], theta)
    return float(ss[0].detach()), float(grad.detach())


rows = []
for gap in (1e-1, 1e-3, 1e-6, 1e-9, 0.):
    val, grad = eigengrad(gap)
    rows.append({'gap': gap, 'strength': val, 'gradient': grad if math.isfinite(grad) else None,
                 'gradient_finite': math.isfinite(grad), 'gap_times_gradient': gap*grad if math.isfinite(grad) else None})
assert all(abs(row['gap_times_gradient'] + 1.6) < 2e-6 for row in rows[:-1])
assert not rows[-1]['gradient_finite']
step = 1e-8
fd = (eigengrad(1e-3, step)[0]-eigengrad(1e-3, -step)[0])/(2*step)
report['eigengrad'] = {'rows': rows, 'finite_difference_gap': 1e-3,
    'finite_difference_step': step, 'finite_difference_gradient': fd,
    'autograd_gradient': rows[1]['gradient'],
    'relative_gradient_error': abs(fd-rows[1]['gradient'])/abs(rows[1]['gradient'])}
assert report['eigengrad']['relative_gradient_error'] < 1e-6

# Conservation of unweighted transition strength is not conservation of f.
ene = np.array([1., 3.])
qq = np.diag([4., 1.])
new_s = np.diag(rot.T @ qq @ rot)
report['redistribution_conservation'] = {
    'strength_total_before': float(np.trace(qq)), 'strength_total_after': float(new_s.sum()),
    'f_total_before': float(C*np.dot(ene, np.diag(qq))),
    'f_total_after': float(C*np.dot(ene, new_s)),
    'does_not_claim_full_TRK_sum_rule': True}
assert abs(report['redistribution_conservation']['f_total_before']-
           report['redistribution_conservation']['f_total_after']) > 1

# E/f at a diagonal H cannot identify Q offdiagonals.
q_other = np.eye(2)
report['operator_nonidentifiability'] = {
    'same_initial_f_max_error': maxerr(C*ene*np.diag(qdeg), C*ene*np.diag(q_other)),
    'different_mixed_strengths_max_error': maxerr(np.diag(rot.T @ qdeg @ rot), np.diag(rot.T @ q_other @ rot)),
    'minimum_eigenvalues_Q': [float(np.linalg.eigvalsh(z).min()) for z in (qdeg, q_other)]}
assert report['operator_nonidentifiability']['same_initial_f_max_error'] == 0
assert report['operator_nonidentifiability']['different_mixed_strengths_max_error'] > .7


def direct_rotation_strength(qin, theta):
    k = torch.stack((torch.stack((theta*0, theta)), torch.stack((-theta, theta*0))))
    uu = torch.matrix_exp(k)
    return torch.diagonal(uu.T @ qin @ uu)


zero = torch.tensor(0., requires_grad=True)
diagonal_grad, = torch.autograd.grad(direct_rotation_strength(torch.diag(torch.tensor([1., 2.])), zero)[0], zero)
zero = torch.tensor(0., requires_grad=True)
coherent_grad, = torch.autograd.grad(direct_rotation_strength(torch.tensor(qdeg), zero)[0], zero)
report['identity_rotation_trainability'] = {'diagonal_Q_rotation_gradient_at_identity': float(diagonal_grad),
    'coherent_Q_rotation_gradient_at_identity': float(coherent_grad)}
assert float(diagonal_grad) == 0 and abs(float(coherent_grad)) > 1


def stable_adapter(factors, ordered_pair_logits, kappa=.1, eps=1e-8):
    """Synthetic minimal empirical proposal, not a production MTO module.

    factors are raw even-tensor C factors (flattened) with norm squared=strength.
    The correlation factor makes the rotation covariant to row-sign gauge.
    No eigenvector, eigengap, sorting, QC feature, or learned calibration occurs.
    """
    gram = factors @ factors.T
    norm = (torch.diagonal(gram)+eps**2).sqrt()
    corr = gram/(norm[:, None]*norm[None, :])
    skew = kappa*corr*torch.tanh(ordered_pair_logits-ordered_pair_logits.T)
    uu = torch.matrix_exp(skew)
    transformed = uu.T @ factors
    return transformed.square().sum(-1), uu, skew


factors = torch.tensor([[1., .4, -.2], [.3, .8, .1], [.2, -.1, .6]])
zero_logits = torch.zeros((3, 3), requires_grad=True)
ss0, uu0, _ = stable_adapter(factors, zero_logits)
g0, = torch.autograd.grad(ss0[0], zero_logits)
assert torch.equal(ss0, factors.square().sum(-1))
assert float(g0.norm()) > 0
logits = torch.tensor([[0., .3, -.1], [-.2, 0., .5], [.7, -.4, 0.]])
ss, uu, kk = stable_adapter(factors, logits)
signs = torch.diag(torch.tensor([1., -1., 1.]))
ss_sign, uu_sign, _ = stable_adapter(signs @ factors, logits)
pp = torch.eye(3)[torch.tensor([2, 0, 1])]
ss_perm, _, _ = stable_adapter(pp @ factors, pp @ logits @ pp.T)
naive_u = torch.matrix_exp(torch.tensor([[0., .1, 0.], [-.1, 0., .1], [0., -.1, 0.]]))
naive1 = (naive_u.T @ factors).square().sum(-1)
naive2 = (naive_u.T @ signs @ factors).square().sum(-1)
z = torch.zeros((3, 3), requires_grad=True)
zs, _, _ = stable_adapter(z, logits)
zg, = torch.autograd.grad(zs.sum(), z)
report['stable_adapter'] = {
    'exact_identity_strength': bool(torch.equal(ss0, factors.square().sum(-1))),
    'identity_pair_logit_gradient_l2': float(g0.norm()),
    'orthogonality_max_error': float((uu.T @ uu-torch.eye(3)).abs().max()),
    'strength_total_error': abs(float(ss.sum()-factors.square().sum())),
    'row_sign_gauge_strength_max_error': float((ss_sign-ss).abs().max()),
    'row_sign_gauge_U_covariance_max_error': float((uu_sign-signs @ uu @ signs).abs().max()),
    'permutation_strength_max_error': float((ss_perm-pp @ ss).abs().max()),
    'naive_fixed_rotation_row_sign_failure_max_error': float((naive1-naive2).abs().max()),
    'zero_factors_forward_and_gradient_finite': bool(torch.isfinite(zs).all() and torch.isfinite(zg).all())}
assert report['stable_adapter']['row_sign_gauge_strength_max_error'] < 1e-12
assert report['stable_adapter']['permutation_strength_max_error'] < 1e-12
assert report['stable_adapter']['naive_fixed_rotation_row_sign_failure_max_error'] > .1

# Without eigendecomposition, gradients do not depend on inverse energy gaps.
report['stable_adapter']['near_gap_gradients'] = []
for gap in (1e-1, 1e-6, 0.):
    param = torch.zeros((3, 3), requires_grad=True)
    ss, _, _ = stable_adapter(factors, param)
    energy = torch.tensor([2-gap/2, 2+gap/2, 3.])
    grad, = torch.autograd.grad((C*energy*ss)[0], param)
    report['stable_adapter']['near_gap_gradients'].append({'gap': gap, 'gradient_l2': float(grad.norm()),
                                                         'finite': bool(torch.isfinite(grad).all())})

# Use native even tensors for Gram factors: O(3)-invariant overlaps, including inversion.
raw = rng.normal(size=(3, 3, 3))
even = (raw + raw.transpose(0, 2, 1))/2
spatial, _ = np.linalg.qr(rng.normal(size=(3, 3)))
gram = even.reshape(3, -1) @ even.reshape(3, -1).T
spatial_errors = []
for rr in (spatial, -spatial, -np.eye(3)):
    changed = rr @ even @ rr.T
    spatial_errors.append(maxerr(changed.reshape(3, -1) @ changed.reshape(3, -1).T, gram))
report['spatial_symmetry'] = {'even_factor_Gram_O3_max_error': max(spatial_errors),
    'centrosymmetric_even_C_strength': float(np.square(np.diag([1., 0., 0.])).sum()),
    'ordinary_polar_vector_fixed_by_inversion_must_be_zero': True,
    'even_factor_rank_bound_for_symmetric_3x3_C': 6,
    'not_a_physical_rank3_dipole_Gram': True}
assert report['spatial_symmetry']['even_factor_Gram_O3_max_error'] < 1e-12

# Row-sign covariance does not identify all latent C square-root choices.
c_native = torch.stack((torch.diag(torch.tensor([1., 2., 0.])),
                        torch.diag(torch.tensor([2., .3, .2]))))
c_alt = c_native.clone()
c_alt[0, 0, 0] = -c_alt[0, 0, 0]
pair = torch.tensor([[0., .4], [-.4, 0.]])
one, _, _ = stable_adapter(c_native.flatten(1), pair)
two, _, _ = stable_adapter(c_alt.flatten(1), pair)
assert torch.equal(c_native @ c_native.transpose(-1, -2), c_alt @ c_alt.transpose(-1, -2))

# A shared ordered-pair network is not automatically covariant under arbitrary
# rotations of an exactly degenerate electronic subspace. With outside-state
# coupling, even the designated subspace sum can depend on that latent choice.
w = torch.eye(3)
w[:2, :2] = torch.tensor(rotation(math.pi/4))
mixed_original, _, _ = stable_adapter(factors, logits)
mixed_represented, _, _ = stable_adapter(w.T @ factors, logits)
subspace_delta = abs(float(mixed_original[:2].sum()-mixed_represented[:2].sum()))
assert subspace_delta > 1e-5
report['minimal_adapter_limits'] = {
    'same_per_state_A_different_latent_C_prediction_max_error': float((one-two).abs().max()),
    'degenerate_subspace_rotation_without_covariant_pair_operator_changes_subspace_total': subspace_delta,
    'all_dark_target_equal_energy_example_strength_total': 5.,
    'minimum_possible_squared_strength_error_from_fixed_total_for_two_dark_states': 12.5,
    'arbitrary_latent_factor_gauge_and_full_degenerate_subspace_gauge_not_solved': True}
assert report['minimal_adapter_limits']['same_per_state_A_different_latent_C_prediction_max_error'] > 1e-3
report['all_tests_passed'] = True
print(json.dumps(report, indent=2, allow_nan=False))
