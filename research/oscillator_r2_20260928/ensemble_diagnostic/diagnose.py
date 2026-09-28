#!/usr/bin/env python3
"""Fixed, validation-only ensemble diagnostic. See PROTOCOL.json."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path('/home/inspur/MTO-1')
OUT = ROOT / 'research/oscillator_r2_20260928/ensemble_diagnostic'
FILES = {
    'mto_eta0': ROOT / 'experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz',
    'mto_eta01': ROOT / 'experiments/qm9s_eta_Ef_20260926/runs/mto_eta01/val_predictions.npz',
    'mto_eta1': ROOT / 'experiments/qm9s_eta_Ef_20260926/runs/mto_eta1/val_predictions.npz',
}
KEYS = ('ids', 'indices', 'f', 'f_true', 'mask_f_true')


def load_one(path):
    with np.load(path, allow_pickle=False) as z:
        absent = sorted(set(KEYS) - set(z.files))
        if absent:
            raise RuntimeError(f'{path}: missing keys {absent}')
        return {k: np.array(z[k], copy=True) for k in KEYS}


def canonical_id(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def score(y, p, mask):
    yy, pp = np.asarray(y)[mask], np.asarray(p)[mask]
    if not yy.size or not np.all(np.isfinite(yy)) or not np.all(np.isfinite(pp)):
        raise RuntimeError('empty or nonfinite masked f values')
    err = pp - yy
    sse = float(np.dot(err, err))
    tss = float(np.dot(yy - yy.mean(), yy - yy.mean()))
    return {'count': int(yy.size), 'sse': sse, 'r2': float(1.0 - sse/tss),
            'rmse': float(np.sqrt(sse/yy.size)), 'mae': float(np.mean(np.abs(err)))}


def fit_simplex(pred, truth, mask):
    # pred shape: (candidate, molecule, state); fit only rows passed in.
    x = np.moveaxis(pred, 0, -1)[mask]
    y = truth[mask]
    if len(y) == 0:
        raise RuntimeError('empty fold training set')
    def objective(w):
        e = x @ w - y
        return float(e @ e / len(y))
    def gradient(w):
        return (2.0 / len(y)) * (x.T @ (x @ w - y))
    n = pred.shape[0]
    res = minimize(objective, np.full(n, 1.0/n), jac=gradient, method='SLSQP',
                   bounds=[(0.0, 1.0)]*n,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w)-1.0,
                                 'jac': lambda w: np.ones_like(w)}],
                   options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
    if not res.success or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f'simplex optimization failed: {res.message}')
    w = np.clip(res.x, 0.0, 1.0)
    w /= w.sum()
    return w


def main():
    arrays = {name: load_one(path) for name, path in FILES.items()}
    names = list(FILES)
    ref = arrays[names[0]]
    for name in names[1:]:
        a = arrays[name]
        for key in ('ids', 'indices', 'f_true', 'mask_f_true'):
            if not np.array_equal(ref[key], a[key]):
                raise RuntimeError(f'ALIGNMENT FAILURE: {key} differs for {name}')
        if not np.array_equal(ref['f'].shape, a['f'].shape):
            raise RuntimeError(f'PREDICTION SHAPE FAILURE: {name}')
    truth = np.asarray(ref['f_true'], dtype=np.float64)
    mask = np.asarray(ref['mask_f_true'], dtype=bool)
    ids = np.asarray(ref['ids'])
    indices = np.asarray(ref['indices'])
    pred = np.stack([np.asarray(arrays[n]['f'], dtype=np.float64) for n in names])
    if truth.ndim != 2 or pred.shape[1:] != truth.shape or mask.shape != truth.shape or ids.ndim not in (1, 2):
        raise RuntimeError(f'expected aligned molecule-by-state predictions with molecule metadata; got truth={truth.shape}, pred={pred.shape}, ids={ids.shape}, indices={indices.shape}, mask={mask.shape}')
    n_mol, n_state = truth.shape
    if indices.shape != ids.shape or ids.shape[0] != n_mol:
        raise RuntimeError(f'molecule metadata shape mismatch: ids={ids.shape}, indices={indices.shape}, molecules={n_mol}')
    if ids.ndim == 1:
        id_values = ids
    elif ids.shape == truth.shape and np.all(ids == ids[:, :1]):
        id_values = ids[:, 0]
    else:
        raise RuntimeError('molecule ID matrix is not constant across state slots')
    if not np.all(np.isfinite(truth[mask])) or not np.all(np.isfinite(pred[:, mask])):
        raise RuntimeError('nonfinite labeled f values')
    n_mol, n_state = truth.shape
    mol_ids = [canonical_id(v) for v in id_values]
    if len(set(mol_ids)) != n_mol:
        raise RuntimeError('duplicate molecule IDs across rows; molecule fold grouping would be ambiguous')
    fold = np.array([int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 5 for s in mol_ids])
    if set(fold.tolist()) != set(range(5)):
        raise RuntimeError(f'not all five frozen folds populated: {np.bincount(fold, minlength=5).tolist()}')

    valid = mask
    residual = pred - truth[None, :, :]
    residual_flat = np.stack([residual[i][valid] for i in range(len(names))])
    corr = np.corrcoef(residual_flat)
    q90 = float(np.quantile(truth[valid], 0.90))
    bright = valid & (truth >= q90)

    candidates = {n: pred[i] for i, n in enumerate(names)}
    model_metrics = {n: score(truth, candidates[n], valid) for n in names}
    fixed_metrics = {}
    fixed_pred = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            label = f'{names[i]}+{names[j]}_equal'
            p = 0.5 * (pred[i] + pred[j])
            fixed_pred[label] = p
            fixed_metrics[label] = score(truth, p, valid)
    p_all = np.mean(pred, axis=0)
    fixed_pred['all_three_equal'] = p_all
    fixed_metrics['all_three_equal'] = score(truth, p_all, valid)

    oof = np.full_like(truth, np.nan, dtype=np.float64)
    fold_weights = []
    for f in range(5):
        held = fold == f
        train_rows = ~held
        w = fit_simplex(pred[:, train_rows, :], truth[train_rows], mask[train_rows])
        oof[held] = np.tensordot(w, pred[:, held, :], axes=(0, 0))
        fold_weights.append({'fold': f, 'train_molecules': int(train_rows.sum()),
                             'heldout_molecules': int(held.sum()),
                             'weights': {n: float(w[i]) for i, n in enumerate(names)}})
    if not np.all(np.isfinite(oof[valid])):
        raise RuntimeError('OOF predictions contain nonfinite labeled values')
    oof_metrics = score(truth, oof, valid)
    oof_bright = score(truth, oof, bright)
    per_state = []
    for s in range(n_state):
        slot = np.zeros_like(valid)
        slot[:, s] = valid[:, s]
        bright_slot = np.zeros_like(valid)
        bright_slot[:, s] = bright[:, s]
        row = {'state_slot': s, 'count': int(valid[:, s].sum()),
               'sse': {n: score(truth, pred[i], slot)['sse'] for i, n in enumerate(names)},
               'sse_all_three_equal': score(truth, p_all, slot)['sse'],
               'sse_oof_simplex': score(truth, oof, slot)['sse'],
               'bright_count': int(bright[:, s].sum()),
               'bright_sse_oof_simplex': score(truth, oof, bright_slot)['sse'] if bright_slot.any() else None}
        per_state.append(row)

    result = {
        'protocol': 'PROTOCOL.json', 'split': 'validation only',
        'source_files': {n: {'path': str(FILES[n]), 'sha256': hashlib.sha256(FILES[n].read_bytes()).hexdigest()} for n in names},
        'alignment': {'exact_ids_indices_raw_truth_and_mask_match': True,
                      'molecules': int(n_mol), 'states_per_molecule': int(n_state),
                      'molecule_id_shape': list(ids.shape), 'source_row_index_shape': list(indices.shape),
                      'state_slot_order': 'matrix columns 0 through 9; raw truth/mask arrays exactly matched',
                      'molecule_id_shape': list(ids.shape), 'source_row_index_shape': list(indices.shape),
                      'state_slot_order': 'matrix columns 0 through 9; raw truth/mask arrays exactly matched',
                      'valid_values': int(valid.sum()), 'unique_molecule_ids': len(set(mol_ids)),
                      'fold_molecule_counts': np.bincount(fold, minlength=5).tolist()},
        'individual_validation_metrics': model_metrics,
        'fixed_uniform_pair_and_all_means_validation_metrics': fixed_metrics,
        'residual_pearson_correlation': {names[i]: {names[j]: float(corr[i,j]) for j in range(len(names))} for i in range(len(names))},
        'simplex_ensemble_oof_metrics': oof_metrics,
        'simplex_ensemble_oof_bright_metrics': {'threshold_q90': q90, **oof_bright},
        'fold_fitted_weights': fold_weights,
        'mean_fold_weights': {n: float(np.mean([r['weights'][n] for r in fold_weights])) for n in names},
        'per_state': per_state,
        'limitations': ['Weights are fit and scored on held-out folds of saved validation predictions, but source checkpoint selection already used validation.',
                        'No test files or test predictions were opened.',
                        'This is conditional exploratory ensemble evidence, not a generalization claim.']
    }
    (OUT / 'ensemble_metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = [
        '# Validation-only MTO eta ensemble diagnostic', '',
        '## Protocol and alignment', '',
        'Candidates were frozen in `PROTOCOL.json` before scores were calculated: the saved MTO eta=0, eta=0.1, and eta=1 validation predictions. Exact molecule ID and source row-index vectors, raw `f_true`, and masks match across all three sources; the ten prediction columns remain in their stored state-slot order. The split has 6,686 molecules × 10 state slots; five deterministic molecule-level folds were assigned by SHA256(ID) modulo five.', '',
        'No test files were opened. c64G1 was excluded because no saved validation array was available; no checkpoint inference was run. Nonsupervised-energy models were excluded.', '',
        '## Results', '',
        '| Candidate | validation R² | SSE | RMSE |', '|---|---:|---:|---:|'
    ]
    for n, m in model_metrics.items(): lines.append(f"| {n} | {m['r2']:.6f} | {m['sse']:.6f} | {m['rmse']:.6f} |")
    for n, m in fixed_metrics.items(): lines.append(f"| {n} | {m['r2']:.6f} | {m['sse']:.6f} | {m['rmse']:.6f} |")
    lines += ['', 'Five-fold molecule-level simplex fit, scored only on held-out validation molecules:', '',
              f"- OOF R²: {oof_metrics['r2']:.6f}; SSE: {oof_metrics['sse']:.6f}; RMSE: {oof_metrics['rmse']:.6f}.",
              f"- Mean fold weights: {json.dumps(result['mean_fold_weights'], sort_keys=True)}.",
              f"- Bright threshold (validation raw truth q90): {q90:.8g}; OOF bright SSE: {oof_bright['sse']:.6f} across {oof_bright['count']} transitions.",
              '', 'Residual Pearson correlations:', '', '| |' + '|'.join(names) + '|', '|' + '---|'*(len(names)+1)]
    for i, n in enumerate(names): lines.append('|' + n + '|' + '|'.join(f'{corr[i,j]:.6f}' for j in range(len(names))) + '|')
    lines += ['', 'Per-state-slot SSE and bright OOF SSE are in `ensemble_metrics.json`.', '',
              '## Interpretation', '',
              'This diagnostic tests whether the saved eta MTO predictions contain enough complementary residual variation to justify a later frozen ensemble evaluation. The reported CV is conditional on fixed predictions: their checkpoint selection already used validation, so the CV scores are exploratory. A later untouched split is needed before claiming improved generalization. The diagnostic does not select or alter a production model.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()