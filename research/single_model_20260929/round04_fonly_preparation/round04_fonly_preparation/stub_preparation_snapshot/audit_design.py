"""Read only calibration design inputs/masks/identities; never decode f_true or E."""
from common import require_cpu
require_cpu()
import hashlib
import json
import zipfile
from unittest.mock import patch
import numpy as np
from common import ROOT, PARENT, ROUND03, settings, sha, read_json, atomic_json, source_hashes, require_preparation_authority
from scalar_map import basis, design_diagnostics, KNOTS


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    require_preparation_authority(); cfg = settings()
    publication = read_json(PARENT/'ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json')
    assert publication['remote_verified'] is True
    assert publication['commit'] == '4f9ae50647f957ac7d68ab1a20200a79fa527cf9'
    previous_manifest = read_json(ROUND03/'FROZEN_MANIFEST.json')
    producer = ROUND03/'affine_stage.py'
    assert sha(producer) == previous_manifest['source_hashes'][str(producer)]
    frozen = read_json(ROUND03/'affine/COEFFICIENTS_FROZEN.json')
    path = ROUND03/'affine/calibration_predictions.npz'
    assert sha(path) == frozen['calibration_arrays_sha256'] == cfg['calibration_array_sha256']
    split = read_json(ROUND03/'split_manifest.json')['arrays']['calibration']
    allowed = {'indices.npy','ids.npy','mask_f.npy','full_baseline_native_f.npy','source_native_f.npy'}
    decoded = []; original_open = zipfile.ZipFile.open
    def guarded_open(self, name, *args, **kwargs):
        member = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if member not in allowed:
            raise PermissionError('Forbidden calibration member: '+str(member))
        decoded.append(member)
        return original_open(self, name, *args, **kwargs)
    with patch.object(zipfile.ZipFile, 'open', guarded_open):
        with np.load(path, allow_pickle=False) as archive:
            indices = archive['indices']; ids = archive['ids']; mask = archive['mask_f']
            assert mask.dtype == np.bool_ and mask.shape == (cfg['calibration_molecules'],10)
            assert len(indices) == len(ids) == cfg['calibration_molecules']
            assert len(np.unique(indices)) == len(indices)
            assert array_sha(indices) == cfg['calibration_index_bytes_sha256'] == split['indices_sha256']
            assert array_sha(ids) == split['ids_sha256']
            assert int(mask.sum()) == cfg['calibration_valid_labels']
            sources = {}
            for name, field in (('in_sample','full_baseline_native_f'),('heldout_source','source_native_f')):
                all_f = archive[field]
                assert all_f.shape == mask.shape and np.isfinite(all_f[mask]).all()
                x = np.asarray(all_f[mask],dtype=np.float64)
                design = basis(x); diagnostics = design_diagnostics(design)
                sources[name] = {'design': diagnostics, 'valid_count': len(x),
                    'prediction_min': float(x.min()), 'prediction_max': float(x.max()),
                    'support_counts': {'below_q90': int((x<KNOTS[0]).sum()),
                        'q90_to_q99': int(((x>=KNOTS[0])&(x<KNOTS[1])).sum()),
                        'at_or_above_q99': int((x>=KNOTS[1]).sum())},
                    'negative_native_count': int((x<0).sum()), 'native_array_sha256': array_sha(all_f)}
                assert sum(sources[name]['support_counts'].values()) == cfg['calibration_valid_labels']
    for item in frozen['maps'].values():
        assert item['count'] == cfg['calibration_valid_labels']
        assert item['zero_target_count'] == cfg['historical_zero_targets_from_pinned_receipt']
    result = {'passed': all(v['design']['passed'] for v in sources.values()),
        'target_free_calibration_design_audit': True, 'actual_target_or_energy_fields_decoded': [],
        'decoded_members': sorted(set(decoded)), 'coefficient_solver_called': False,
        'model_inference': False, 'outer_validation_or_test_access': False,
        'same_rows_and_masks_for_both_sources': True, 'valid_labels': int(mask.sum()),
        'indices_sha256': array_sha(indices), 'ids_sha256': array_sha(ids),
        'mask_sha256': array_sha(mask),
        'zero_target_count_from_published_receipt_only': cfg['historical_zero_targets_from_pinned_receipt'],
        'zero_target_count_newly_recomputed': False, 'sources': sources,
        'input_hashes': {str(path): sha(path), str(producer): sha(producer),
            str(ROUND03/'affine/COEFFICIENTS_FROZEN.json'): sha(ROUND03/'affine/COEFFICIENTS_FROZEN.json'),
            str(ROUND03/'split_manifest.json'): sha(ROUND03/'split_manifest.json')},
        'source_hashes': source_hashes(['audit_design.py','scalar_map.py','common.py','settings.json'])}
    atomic_json(result, ROOT/'DESIGN_AUDIT.json')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Prespecified design gate failed; no fallback permitted')


if __name__ == '__main__':
    main()
