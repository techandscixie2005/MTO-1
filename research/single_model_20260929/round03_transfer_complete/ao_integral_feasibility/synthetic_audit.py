"""Small analytic Gaussian/AO-pair algebra audit; no QC package/data/model fit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260930)


def contraction(d, t):
    return np.einsum('aij,lji->la', d, t)


def tensor(d, t):
    mu = contraction(d, t)
    return mu.T @ mu


def err(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    # One normalized primitive s and its three p partners at the same center.
    # s=(2a/pi)^.75 exp(-a*r^2); p_i=2sqrt(a)*r_i*s.
    # S=I and <s|r_i|p_j>=delta_ij/(2sqrt(a)) analytically.
    alpha = .8
    m = 4
    overlap = np.eye(m)
    dipole = np.zeros((3, m, m))
    for a in range(3):
        dipole[a, 0, a + 1] = dipole[a, a + 1, 0] = 1 / (2 * np.sqrt(alpha))
    t = np.zeros((3, m, m))
    for a in range(3):
        t[a, 0, a + 1] = t[a, a + 1, 0] = .5
    # A single local s->p_z transition has a nonzero z moment at a nucleus at0.
    onsite_mu = contraction(dipole, t[2:3])[0]
    onsite_a = tensor(dipole, t[2:3])
    aggregate_a = tensor(dipole, t)
    assert np.allclose(aggregate_a, np.eye(3) / (4 * alpha), atol=1e-14)

    origin_shift = np.array([.7, -1.2, 2.])
    shifted = dipole - origin_shift[:, None, None] * overlap
    origin_error = err(tensor(shifted, t), aggregate_a)
    vec_t = t.reshape(3, -1)
    covariance = vec_t.T @ vec_t
    neutrality = float(np.linalg.norm(covariance @ overlap.ravel()))
    min_eig = float(np.linalg.eigvalsh(covariance).min())
    inversion_ao = np.diag([1., -1., -1., -1.])
    inverted_t = np.einsum('ij,ljk,kn->lin', inversion_ao, t, inversion_ao)
    assert np.allclose(inverted_t, -t)
    covariance_inversion_error = err(inverted_t.reshape(3, -1).T @ inverted_t.reshape(3, -1), covariance)

    # Joint Cartesian and AO representation changes, including reflection and
    # a complete AO reordering. D and contravariant coefficients both transform.
    joint_errors = []
    for determinant in (1, -1):
        cart, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
        cart[:, 0] *= determinant / np.linalg.det(cart)
        basis, _ = np.linalg.qr(RNG.normal(size=(m, m)))
        rotated_d = np.einsum('ab,ij,bjk,kn->ain', cart, basis.T, dipole, basis)
        transformed_t = np.einsum('ij,ljk,kn->lin', basis.T, t, basis)
        joint_errors.append(err(tensor(rotated_d, transformed_t), cart @ aggregate_a @ cart.T))
    permutation = np.eye(m)[:, [2, 0, 3, 1]]
    perm_d = np.einsum('ij,ajk,kn->ain', permutation.T, dipole, permutation)
    perm_t = np.einsum('ij,ljk,kn->lin', permutation.T, t, permutation)
    permutation_error = err(tensor(perm_d, perm_t), aggregate_a)

    # Nonorthogonal AO change chi'=chi*B: S'=B^T*S*B,
    # D'=B^T*D*B, T'=B^-1*T*B^-T. Ordinary trace(T')=0 is WRONG.
    b = np.array([[1.2, .4, .1, 0.], [.2, .8, 0., .1], [0., .1, 1.4, .3], [.1, 0., .2, .7]])
    bi = np.linalg.inv(b)
    sp = b.T @ overlap @ b
    dp = np.einsum('ij,ajk,kn->ain', b.T, dipole, b)
    tp = np.einsum('ij,ljk,kn->lin', bi, t, bi.T)
    basis_error = err(tensor(dp, tp), aggregate_a)
    correct_charge = float(np.max(np.abs(np.einsum('lij,ji->l', tp, sp))))
    wrong = np.diag([1., -1., 0., 0.])[None]
    wrong_charge = float(np.einsum('lij,ji->l', wrong, sp)[0])
    wrong_origin_drift = err(tensor(dp - origin_shift[:, None, None] * sp, wrong), tensor(dp, wrong))
    assert abs(wrong_charge) > .1 and wrong_origin_drift > .1

    # Rebuilding an Euclidean neutral identity covariance in each arbitrary AO
    # basis is also not equivalent to transporting the learned covariance.
    def euclidean_projector(s):
        v = s.ravel()
        return np.eye(len(v)) - np.outer(v, v) / (v @ v)
    def from_cov(d, k):
        v = d.reshape(3, -1)
        return v @ k @ v.T
    naive_basis_change = err(from_cov(dp, euclidean_projector(sp)), from_cov(dipole, euclidean_projector(overlap)))
    assert naive_basis_change > .01

    # Coherent AO-pair cancellation is retained by PSD covariance cross terms.
    pair_d = np.array([1., 1.])
    pair_t = np.array([1., -1.])
    pair_k = np.outer(pair_t, pair_t)
    coherent_strength = float(pair_d @ pair_k @ pair_d)
    diagonal_only_strength = float(pair_d @ np.diag(np.diag(pair_k)) @ pair_d)
    assert coherent_strength == 0 and diagonal_only_strength == 2

    # A neutral dark AO component changes K but leaves all dipole observables.
    dark = np.diag([1., -1., 0., 0.])
    altered = t[2:3] + dark[None]
    invisible_error = err(tensor(dipole, altered), onsite_a)
    k_difference = float(np.linalg.norm(altered.reshape(1, -1).T @ altered.reshape(1, -1) - np.outer(t[2].ravel(), t[2].ravel())))
    sign_error = err(tensor(dipole, -t[2:3]), onsite_a)
    # Electronic rotations inside an exact two-state degenerate subspace alter
    # individual A, but preserve its group sum. Individual K cannot fix gauge.
    theta = np.pi / 4
    u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    electronic_rotated = np.einsum('lk,kij->lij', u, t[:2])
    group_error = err(tensor(dipole, electronic_rotated), tensor(dipole, t[:2]))
    individual_change = err(tensor(dipole, electronic_rotated[:1]), tensor(dipole, t[:1]))
    assert individual_change > .1

    # Unique overlap matrix functions avoid a chosen eigenvector gauge, but
    # near linear dependencies amplify errors and require a declared policy.
    overlap_eigenvalues = np.array([1., 1., 1e-12])
    inverse_root_norm = float(np.max(overlap_eigenvalues ** -.5))
    result = {
        'passed': True, 'scope': 'analytic Gaussian toy plus AO-pair algebra, not target-basis or dataset reconstruction',
        'no_model_dataset_GPU_install_or_fitting': True,
        'primitive_exponent_bohr_minus2': alpha,
        'onsite': {'mu_z': onsite_mu.tolist(), 'A_zz': float(onsite_a[2, 2]),
                   'nuclear_point_charge_span_dimension': 0,
                   'isotropic_three_component_aggregate_A': aggregate_a.tolist(),
                   'single_atom_degeneracy_caveat': 'Individual directional transition is a local algebra example; an isolated spherical atom cannot choose a physical unique direction from geometry.'},
        'PSD_and_origin': {'min_covariance_eigenvalue': min_eig, 'K_times_vecS_norm': neutrality,
                           'origin_shift_max_A_error': origin_error},
        'symmetry': {'proper_and_improper_joint_transformation_max_errors': joint_errors,
                     'AO_permutation_error': permutation_error,
                     'inversion_T_sign_change_covariance_error': covariance_inversion_error,
                     'electronic_state_phase_flip_A_error': sign_error},
        'nonorthogonal_basis': {'transported_coefficients_A_error': basis_error,
                               'correct_overlap_charge_max_abs': correct_charge,
                               'naive_trace_zero_overlap_charge': wrong_charge,
                               'naive_trace_zero_origin_A_drift': wrong_origin_drift,
                               'recomputed_Euclidean_projector_basis_A_drift': naive_basis_change},
        'coherence': {'PSD_cross_terms_strength': coherent_strength, 'diagonal_only_strength': diagonal_only_strength},
        'unidentified_density': {'neutral_dark_component_A_error': invisible_error, 'covariance_Frobenius_change': k_difference},
        'exact_electronic_degeneracy': {'individual_A_change': individual_change, 'group_sum_A_error': group_error},
        'overlap_conditioning': {'smallest_eigenvalue': 1e-12, 'inverse_square_root_spectral_norm': inverse_root_norm},
        'cost_scaling': {'dense_pair_covariance': 'O(M^4)', 'rank_r_factors': 'O(r*M^2)',
                         'illustrative_M_100_real_symmetric_pair_dimension': 5050,
                         'illustrative_dense_FP64_bytes_per_state': 5050 * 5050 * 8},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (ROOT / 'SYNTHETIC_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    metadata_path = Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925/dataset_manifest.json')
    metadata = json.loads(metadata_path.read_text())
    keys = ['route_section_distribution', 'program_version_distribution', 'rpa_flag_distribution',
            'coordinate_frame_distribution', 'charge_multiplicity_distribution', 'units', 'connectivity']
    record = {'source_path': str(metadata_path), 'source_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'metadata': {k: metadata[k] for k in keys}, 'pyscf_installed_in_pinned_runtime': importlib.util.find_spec('pyscf') is not None,
              'exact_basis_coefficients_order_cartesian_spherical_cross_program_parity_audited': False,
              'raw_labels_logs_or_geometry_arrays_read': False}
    (ROOT / 'METADATA_AUDIT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
