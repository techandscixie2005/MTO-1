"""Bounded TRAIN-only label reconstruction audit; no model or GPU imports."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')
DATA = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data')
SAMPLE_SIZE = 4096
SAMPLE_SEED = 20260930
HARTREE_EV = 27.211386245988


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def selected_rows(archive, name, rows):
    """Decode only selected rows, skipping compressed nonselected byte ranges.

    ZIP decompression must pass preceding bytes, but those rows are never
    interpreted as numbers or retained in an array. Molecule IDs are metadata.
    """
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported NPY version')
        assert not fortran and not dtype.hasobject
        width = int(np.prod(shape[1:]))
        row_bytes = width * dtype.itemsize
        origin = stream.tell()
        values = []
        for index in rows:
            assert 0 <= index < shape[0]
            stream.seek(origin + int(index) * row_bytes)
            raw = stream.read(row_bytes)
            assert len(raw) == row_bytes
            values.append(np.frombuffer(raw, dtype=dtype).copy().reshape(shape[1:]))
        return np.stack(values)


def summary(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not a.size:
        return {'count': 0}
    assert np.isfinite(a).all()
    return {'count': int(a.size), 'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.quantile(a, .9)), 'p99': float(np.quantile(a, .99))}


def main():
    assert sha(DATA / 'splits.json') == '141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca'
    # Only index/ID metadata are loaded from the curated dataset.
    with np.load(DATA / 'dataset.npz', allow_pickle=False) as data:
        train = data['train'].copy()
        ids = data['ids']
        train_ids = ids[train].copy()
    assert len(train_ids) == 120355 and len(np.unique(train_ids)) == len(train_ids)
    selected = np.sort(np.random.default_rng(SAMPLE_SEED).choice(train_ids, SAMPLE_SIZE, replace=False))
    assert np.isin(selected, train_ids).all()
    names = ('energy_eV', 'oscillator_strength', 'oscillator_strength_velocity',
             'velocity_dipole_au', 'state_mask', 'scalar_label_mask',
             'normal_termination', 'response_convergence_reported',
             'unambiguous_single_calculation', 'geometry_present')
    chunks = {name: [] for name in names}
    found = []
    provenance = {}
    for path in sorted((SOURCE / 'arrays').glob('part-*.npz')):
        # IDs only: no label decoding until selected TRAIN positions are known.
        with np.load(path, allow_pickle=False) as data:
            shard_ids = data['molecule_id'].copy()
        rows = np.flatnonzero(np.isin(shard_ids, selected))
        if not len(rows):
            continue
        with zipfile.ZipFile(path) as archive:
            for name in names:
                assert name + '.npy' in archive.namelist(), 'Required field unavailable: ' + name
                chunks[name].append(selected_rows(archive, name, rows))
        found.extend(shard_ids[rows].tolist())
        provenance[str(path)] = sha(path)
    assert np.array_equal(np.sort(found), selected), 'Selected TRAIN IDs missing or duplicated'
    x = {name: np.concatenate(parts) for name, parts in chunks.items()}
    for name in ('normal_termination', 'response_convergence_reported',
                 'unambiguous_single_calculation', 'geometry_present'):
        assert x[name].all(), 'Selected TRAIN source quality flag missing: ' + name
    E, fL, fV, p = [x[name] for name in ('energy_eV', 'oscillator_strength',
                                       'oscillator_strength_velocity', 'velocity_dipole_au')]
    state = x['state_mask'].astype(bool)
    finite_E = np.isfinite(E) & (E > 0)
    finite_p = np.isfinite(p).all(-1)
    finite_fV = np.isfinite(fV)
    finite_fL = np.isfinite(fL)
    available = state & finite_E & finite_p & finite_fV
    assert available.any(), 'No verified optional velocity labels in TRAIN sample'
    # Never replace NaN with zero; present printed zeros remain actual values.
    omega = E[available] / HARTREE_EV
    p2 = np.square(p[available]).sum(-1)
    reconstructed = (2.0 / 3.0) * p2 / omega
    residual = reconstructed - fV[available]
    # Audit the printed four-decimal grid before interpreting rounding bounds.
    grid = {name: float(np.abs(value * 1e4 - np.rint(value * 1e4)).max())
            for name, value in [('energy_eV', E[available]), ('velocity_components', p[available]),
                                ('printed_velocity_f', fV[available])]}
    assert max(grid.values()) < 1e-7, 'Four-decimal rounding assumption is not supported'
    delta = 5e-5
    p_lower = np.square(np.maximum(np.abs(p[available]) - delta, 0)).sum(-1)
    p_upper = np.square(np.abs(p[available]) + delta).sum(-1)
    f_lower = (2 / 3) * p_lower / ((E[available] + delta) / HARTREE_EV)
    f_upper = (2 / 3) * p_upper / ((E[available] - delta) / HARTREE_EV)
    rounding_overlap = ((f_lower <= fV[available] + delta + 1e-12) &
                        (f_upper >= fV[available] - delta - 1e-12))
    comparable = available & finite_fL
    discrepancy = fV[comparable] - fL[comparable]
    positive_length = comparable & (fL >= 1e-3)
    sst = np.square(fL[comparable] - fL[comparable].mean()).sum()
    report = {
        'status': 'bounded_train_label_audit_complete', 'sample_size_molecules': SAMPLE_SIZE,
        'sample_seed': SAMPLE_SEED, 'sampling': 'uniform_without_replacement_from_frozen_train_IDs',
        'sample_IDs_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
        'sample_ID_dtype': str(selected.dtype), 'all_sample_IDs_in_frozen_train': True,
        'decoded_validation_label_rows': 0, 'decoded_test_label_rows': 0,
        'compressed_byte_note': 'NPZ streams skip nonselected bytes without decoding them as label rows',
        'model_inference': False, 'gpu_used': False, 'raw_logs_loaded': False,
        'optional_missing_policy': 'extractor initializes NaN; presence requires finite components/fV, never zero fill',
        'coverage': {'state_slots': int(state.size), 'present_state_labels': int(state.sum()),
            'scalar_length_mask_true': int((state & x['scalar_label_mask']).sum()),
            'finite_positive_energy': int((state & finite_E).sum()),
            'finite_velocity_vectors': int((state & finite_p).sum()),
            'finite_velocity_strength': int((state & finite_fV).sum()),
            'velocity_reconstruction_pairs': int(available.sum()),
            'finite_length_strength': int((state & finite_fL).sum()),
            'all_zero_velocity_vectors': int((p2 == 0).sum()),
            'printed_zero_velocity_strength': int((fV[available] == 0).sum()),
            'negative_velocity_strength': int((fV[available] < 0).sum())},
        'energy_eV': summary(E[state & finite_E]),
        'formula': 'fV=(2/3)*sum(p_AU**2)/(E_eV/27.211386245988)',
        'velocity_reconstruction': {'signed_residual': summary(residual),
            'absolute_residual': summary(np.abs(residual)),
            'rmse': float(np.sqrt(np.square(residual).mean())),
            'absolute_residual_le_1e_minus4_count': int((np.abs(residual) <= 1e-4).sum()),
            'four_decimal_grid_max_error_after_scale': grid,
            'rounding_interval_overlap_count': int(rounding_overlap.sum()),
            'rounding_interval_failure_count': int((~rounding_overlap).sum()),
            'rounding_bound_assumption': 'Independent plus/minus5e-5 rounding of E_eV, each velocity component, and printed fV'},
        'velocity_vs_length': {'signed_difference': summary(discrepancy),
            'absolute_difference': summary(np.abs(discrepancy)),
            'rmse': float(np.sqrt(np.square(discrepancy).mean())),
            'agreement_r2_not_model_accuracy': float(1 - np.square(discrepancy).sum() / sst),
            'relative_abs_difference_length_at_least_1e_minus3':
                summary(np.abs((fV[positive_length] - fL[positive_length]) / fL[positive_length])),
            'ratio_subset_note': 'Threshold only avoids unstable relative denominators; all present labels use absolute diagnostics'},
        'limitations': ['Sample coverage does not establish full TRAIN coverage',
            'No independent Gaussian imaginary/sign convention audit from original logs',
            'Norm reconstruction cannot identify electronic phase or vector-gauge alignment',
            'No duplicate/ambiguous optional-table audit beyond existing parser',
            'No magnetic-label analysis or new auxiliary objective'],
        'primary_reference': 'https://pyscf.org/_modules/pyscf/tdscf/rhf.html',
        'source_hashes': {str(SOURCE / name): sha(SOURCE / name) for name in
                          ('extract_qm9s.py', 'dataset_manifest.json', 'README.md', 'validation_report.json')},
        'array_shard_hashes': provenance, 'code_sha256': sha(Path(__file__)),
    }
    (ROOT / 'VELOCITY_TRAIN_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('coverage', 'velocity_reconstruction', 'velocity_vs_length')}))


if __name__ == '__main__':
    main()
