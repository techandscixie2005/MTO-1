"""Small synthetic mathematical/gate checks; no file-backed training data or model."""
from common import require_cpu
require_cpu()
import copy
import json
import numpy as np
from common import ROOT, atomic_json, require_preparation_authority, source_hashes
from scalar_map import SCALE, KNOTS, basis, apply_map, affine_parameters, solve_synthetic, require_design
from execution_gate import check_bindings, verify_execution


def rejected(function, kind):
    try:
        function()
    except kind:
        return True
    raise AssertionError('Expected rejection did not occur')


def main():
    require_preparation_authority()
    x = np.array([0., .01, .03, .0549, .08, .15, .2412, .3, .6, 1.2])
    known = np.array([-.002, .047, -.013, .009])  # Hand-set synthetic constants.
    y = basis(x) @ known
    solved, diagnostics = solve_synthetic(x, y, fixture_name='synthetic_known_piecewise_linear')
    recovery = float(np.max(np.abs(solved-known)))
    assert recovery < 1e-12
    assert np.max(np.abs(basis(x) @ solved-y)) < 1e-12
    nesting = []
    for alpha, beta in ((1.0180950011215777, -.0002909346448593461),
                        (.9025853252242239, .0018285582112617521)):
        actual = apply_map(x, affine_parameters(alpha, beta))
        expected = np.maximum(0, alpha*x+beta)
        delta = float(np.max(np.abs(actual-expected)))
        assert delta < 1e-12
        nesting.append(delta)
    # Evaluate both algebraic pieces at the exact knot; no finite-difference tolerance is needed.
    k0, k1 = KNOTS
    left0 = known[0] + known[1]*k0/SCALE
    right0 = left0 + known[2]*(k0-k0)/SCALE
    left1 = known[0] + known[1]*k1/SCALE + known[2]*(k1-k0)/SCALE
    right1 = left1 + known[3]*(k1-k1)/SCALE
    assert left0 == right0 and left1 == right1
    upper = np.array([.4, .8, 1.2]); raw = basis(upper) @ known
    assert abs(float(raw[2]-2*raw[1]+raw[0])) < 1e-12
    assert apply_map(np.array([0.]), known)[0] == 0
    failures = {
        'nonfinite_prediction': rejected(lambda: basis([np.nan]), ValueError),
        'nonfinite_target': rejected(lambda: solve_synthetic(x, np.full_like(x, np.nan), fixture_name='synthetic_nan'), ValueError),
        'rank_deficient': rejected(lambda: require_design(basis(np.full(8, .1))), ValueError),
        'ill_conditioned_full_rank': rejected(lambda: require_design(np.diag([1., 1., 1., 1e-9])), ValueError),
        'too_few_rows': rejected(lambda: require_design(basis([0., .1, .3])), ValueError),
        'invalid_coefficient_count': rejected(lambda: apply_map(x, [1,2]), ValueError),
        'non_synthetic_solve_request': rejected(lambda: solve_synthetic(x, y, fixture_name='real'), PermissionError),
        'real_dataset_sized_solve': rejected(lambda: solve_synthetic(np.ones(129), np.ones(129), fixture_name='synthetic_invalid_size'), ValueError),
        'absent_production_authority': rejected(lambda: verify_execution(ROOT/'NO_EXECUTION_AUTHORITY.json', ROOT/'NO_MANIFEST.json', ROOT/'NO_REVIEW.json', ROOT/'NO_PUBLICATION.json'), PermissionError)}
    auth = dict(phase='production_execution', real_target_fit_authorized=True,
                fixed_validation_authorized=True, test_access_authorized=False,
                source_manifest_sha256='m', independent_review_sha256='r', publication_receipt_sha256='p',
                maximum_real_data_solves=2, maximum_validation_comparisons=1)
    review = dict(passed=True, source_manifest_sha256='m')
    publication = dict(remote_verified=True, download_before_stage_before_commit_push=True,
                       source_manifest_sha256='m', independent_review_sha256='r')
    check_bindings(auth, 'm', 'r', 'p', review, publication)
    for name, value in [('phase', 'preparation_only'), ('source_manifest_sha256','wrong'),
                        ('independent_review_sha256','wrong'), ('publication_receipt_sha256','wrong'),
                        ('maximum_real_data_solves', 3), ('test_access_authorized', True)]:
        altered = copy.deepcopy(auth); altered[name] = value
        failures['execution_binding_'+name] = rejected(lambda: check_bindings(altered, 'm','r','p',review,publication), PermissionError)
    for name in ('remote_verified', 'download_before_stage_before_commit_push'):
        altered = copy.deepcopy(publication); altered[name] = False
        failures['publication_'+name] = rejected(lambda: check_bindings(auth,'m','r','p',review,altered), PermissionError)
    result = {'passed': True, 'synthetic_only': True, 'real_target_solve': False,
              'array_or_checkpoint_reads': False, 'known_recovery_max_abs': recovery,
              'synthetic_design': diagnostics, 'affine_nesting_max_abs': nesting,
              'continuity_and_linear_tail': True, 'clamp_check': True, 'failure_checks': failures,
              'source_hashes': source_hashes(['synthetic_checks.py','scalar_map.py','execution_gate.py','common.py','settings.json'])}
    atomic_json(result, ROOT/'SYNTHETIC_CHECKS.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
