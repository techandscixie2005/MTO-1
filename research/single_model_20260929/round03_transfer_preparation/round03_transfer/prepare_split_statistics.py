"""Create the audited internal-TRAIN split and fit-only normalization; CPU only.

Index arrays stay server-only. Compressed nonselected bytes may be passed while
streaming NPZ members; no non-fit target row is decoded or retained.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
SOURCE = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
DATA = SOURCE / 'data'
IDENTITY = Path('/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json')
sys.path.insert(0, str(CAMPAIGN / 'clean_source_feasibility'))
sys.path.insert(0, str(CAMPAIGN / 'reports/velocity_audit'))
from audit_clean_source import grouped_split
from audit_velocity_labels import selected_rows

EXPECTED = {
    'identity': '9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4',
    'splits': '141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca',
    'dataset': 'be8fadca203429575d70642b617730592693be40858dca7189c098a671596330',
    'raw_labels': '621dc8723fb6dc96681851a26e4f52fcb7747a4778e57a249a75d22f3af4c632',
    'fit_indices': 'b7b00dfe514ae57f3dd609a60dc259625007a5f43288ff75e2c83a3a7e71da37',
    'calibration_indices': '9547ca89feff3301f5e4c6fcd736be7a9186878a9fcb1fec8695c5c12d60707b',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def write_indices(path, values):
    if path.exists():
        assert np.array_equal(np.load(path, allow_pickle=False), values), 'Refuse different existing split'
    else:
        temp = path.with_suffix('.npy.tmp')
        with open(temp, 'wb') as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temp, path)


def main():
    assert str(ROOT) == '/home/inspur/MTO-1/research/single_model_20260929/round03_transfer'
    provenance = {'identity': IDENTITY, 'splits': DATA / 'splits.json',
                  'dataset': DATA / 'dataset.npz', 'raw_labels': DATA / 'raw_labels.npz'}
    for key, path in provenance.items():
        assert sha(path) == EXPECTED[key], 'Changed source: ' + key
    identity = json.loads(IDENTITY.read_text())
    with np.load(DATA / 'dataset.npz', allow_pickle=False) as archive:
        train = archive['train'].copy()
        ids = archive['ids'].copy()
    assert all(int(ids[i]) == int(row[0]) for i, row in enumerate(identity))
    fit, calibration, group_metadata = grouped_split(train, identity, seed=20260930)
    assert len(fit) == 96284 and len(calibration) == 24071
    assert array_sha(fit) == EXPECTED['fit_indices']
    assert array_sha(calibration) == EXPECTED['calibration_indices']
    assert np.array_equal(np.sort(np.concatenate([fit, calibration])), np.sort(train))
    (ROOT / 'data').mkdir(exist_ok=True)
    arrays = {}
    for key, values in [('fit', fit), ('calibration', calibration)]:
        path = ROOT / 'data' / (key + '_indices.npy')
        write_indices(path, values)
        arrays[key] = {'path': str(path), 'sha256': sha(path), 'indices_sha256': array_sha(values),
                       'molecules': len(values), 'dtype': str(values.dtype),
                       'ids_sha256': array_sha(ids[values]), 'archive_allowed': False}

    selected = np.sort(fit)
    with zipfile.ZipFile(DATA / 'dataset.npz') as archive:
        atom_z = selected_rows(archive, 'z', selected)
    with zipfile.ZipFile(DATA / 'raw_labels.npz') as archive:
        raw_ids = selected_rows(archive, 'ids', selected)
        energy = selected_rows(archive, 'E', selected).astype(np.float64)
        matrix = selected_rows(archive, 'A', selected).astype(np.float64)
        mask_e = selected_rows(archive, 'mask_E', selected).astype(bool)
        mask_a = selected_rows(archive, 'mask_A', selected).astype(bool)
    assert np.array_equal(raw_ids, ids[selected])
    assert energy.shape == (96284, 10) and matrix.shape == (96284, 10, 3, 3)
    assert mask_e.all() and mask_a.all(), 'Frozen source expects all valid E/A; no implicit exclusion'
    assert np.isfinite(energy).all() and np.isfinite(matrix).all()
    energy_mean = energy.mean(axis=0)
    s_e2 = float(np.square(energy - energy_mean).mean())
    s_a2 = float(np.square(matrix).sum(axis=(-2, -1)).mean())
    atom_counts = (atom_z != 0).sum(axis=1)
    n_ref = float(np.median(atom_counts))
    assert s_e2 > 0 and s_a2 > 0 and n_ref > 0
    # Independent moment identity checks the reduction and FP64 convention.
    moment_e2 = float((np.square(energy).mean(axis=0) - energy_mean ** 2).mean())
    block_a2 = float(sum(np.einsum('nsij,nsij->', block, block) for block in np.array_split(matrix, 31)) / mask_a.sum())
    assert abs(moment_e2 - s_e2) < 1e-11 and abs(block_a2 - s_a2) < 1e-12
    normalization = {
        'sE2': s_e2, 'sA2': s_a2, 'E_state_mean': energy_mean.tolist(), 'n_ref': n_ref,
        'source': 'round03_source_fit_only_96284',
        'definition_E': 'mean_na((E_fit_na-mean_n_fit(E_fit_na))^2)',
        'definition_A': 'mean_na(sum_ij(A_fit_naij^2))',
        'definition_n_ref': 'median fit-only atom count',
        'raw_label_reduction_dtype': 'float64', 'fit_indices_sha256': array_sha(fit),
    }
    atomic_json(ROOT / 'fit_normalization.json', normalization)
    split_manifest = {
        'passed': True, 'seed': 20260930, 'arrays': arrays, 'group_metadata': group_metadata,
        'policy': 'shuffle sorted resolved original TRAIN groups; whole-group prefix until 24071 held rows; unresolved groups fit-only; original TRAIN order within each subset',
        'all_original_train_rows_assigned_once': True, 'no_group_broken': True,
        'outer_splits_unchanged': True, 'labels_used_for_split': False,
        'source_hashes': {str(path): EXPECTED[key] for key, path in provenance.items()},
    }
    atomic_json(ROOT / 'split_manifest.json', split_manifest)
    code_paths = [Path(__file__), CAMPAIGN / 'clean_source_feasibility/audit_clean_source.py',
                  CAMPAIGN / 'reports/velocity_audit/audit_velocity_labels.py',
                  SOURCE / 'model_factory.py', SOURCE / 'frozen_reference/models_ea.py']
    audit = {
        'passed': True, 'fit_molecules': len(fit), 'fit_valid_E_labels': int(mask_e.sum()),
        'fit_valid_A_labels': int(mask_a.sum()), 'decoded_target_rows': len(fit),
        'decoded_target_members': ['E', 'A', 'mask_E', 'mask_A'],
        'decoded_calibration_validation_test_target_rows': 0, 'raw_f_decoded': False,
        'all_fit_target_masks_valid': True, 'no_model_fit_or_GPU': True,
        'initialization_requirement': 'fresh torch seed11 MTOEA(config, fit_normalization); do not load initial_model.pt or full-TRAIN target statistics',
        'target_statistics': normalization,
        'independent_formula_errors': {'energy_second_moment_abs': abs(moment_e2 - s_e2), 'A_block_sum_abs': abs(block_a2 - s_a2)},
        'source_hashes': {str(path): sha(path) for path in code_paths + list(provenance.values())},
        'outputs': {name: sha(ROOT / name) for name in ['split_manifest.json', 'fit_normalization.json']},
        'compressed_byte_note': 'Nonselected compressed bytes are skipped without decoding labels; only fitting rows enter reductions.',
    }
    atomic_json(ROOT / 'STATISTICS_AUDIT.json', audit)
    print(json.dumps({'passed': True, 'counts': group_metadata, 'sE2': s_e2, 'sA2': s_a2, 'n_ref': n_ref}))


if __name__ == '__main__':
    main()
