"""Fixed four-column basis and exact least squares; production callers require execution gates."""
import numpy as np

SCALE = 0.050109692115345626
KNOTS = np.array([0.0549, 0.2412], dtype=np.float64)
RCOND = 1e-12
MAX_CONDITION = 1e8


def basis(prediction):
    f = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(f).all():
        raise ValueError('Nonfinite scalar prediction')
    return np.stack((np.ones_like(f), f / SCALE,
                     np.maximum(0.0, (f - KNOTS[0]) / SCALE),
                     np.maximum(0.0, (f - KNOTS[1]) / SCALE)), axis=-1)


def design_diagnostics(design):
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or matrix.shape[0] < 4:
        raise ValueError('Design must have at least four rows and exactly four columns')
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite design')
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > RCOND * singular[0])) if singular[0] else 0
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    passed = rank == 4 and condition is not None and condition <= MAX_CONDITION
    return {'rows': len(matrix), 'columns': 4, 'rank': rank,
            'singular_values': singular.tolist(), 'condition_number': condition,
            'rcond': RCOND, 'maximum_condition_number': MAX_CONDITION, 'passed': passed}


def require_design(design):
    diagnostics = design_diagnostics(design)
    if not diagnostics['passed']:
        raise ValueError('Rank or condition gate failed: %s' % diagnostics)
    return diagnostics


def solve_synthetic(prediction, truth, *, fixture_name):
    """Small explicitly synthetic fixtures for preparation tests."""
    if not isinstance(fixture_name, str) or not fixture_name.startswith('synthetic_'):
        raise PermissionError('Preparation only exposes explicitly synthetic fixtures')
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) > 128:
        raise ValueError('Synthetic fixture must be paired scalar vectors of <=128 rows')
    return solve_fixed(x, y)


def solve_fixed(prediction, truth):
    """Real-data use is exclusively through the independently authorized production entry point."""
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError('Paired scalar vectors required')
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite target')
    design = basis(x)
    diagnostics = require_design(design)
    theta, residuals, rank, singular = np.linalg.lstsq(design, y, rcond=RCOND)
    if rank != 4 or not np.isfinite(theta).all():
        raise ValueError('Invalid least-squares result')
    return theta, diagnostics


def apply_map(prediction, theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (4,) or not np.isfinite(theta).all():
        raise ValueError('Exactly four finite coefficients required')
    raw = basis(prediction) @ theta
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite map output')
    return np.maximum(0.0, raw)


def affine_parameters(alpha, beta):
    return np.array([beta, alpha * SCALE, 0.0, 0.0], dtype=np.float64)
