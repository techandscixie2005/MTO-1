"""Independent synthetic reader boundary checks; no real corpus or label reads."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('OMP_NUM_THREADS') == '2' and os.environ.get('MKL_NUM_THREADS') == '2'
import io, json, tempfile, zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import partition_data as reader
from common import ROOT, sha, immutable_json

def rejects(call, error=AssertionError):
    try:
        call()
    except error:
        return
    raise AssertionError('Expected rejection')

def main():
    checks = {}
    with patch.object(reader, 'sha', return_value='0'*64), patch.object(reader, 'read', side_effect=RuntimeError('Unexpected metadata read')), patch.object(np, 'load', side_effect=RuntimeError('Unexpected array read')):
        rejects(lambda: reader.Partition('train'))
    checks['split_file_pin_mismatch_rejects_before_metadata_or_arrays'] = True
    def matching_sha(path):
        return reader.PINS[Path(path).name]
    def wrong_binding(path):
        return {'passed': True, 'split_manifest_sha256': '0'*64}
    with patch.object(reader, 'sha', side_effect=matching_sha), patch.object(reader, 'read', side_effect=wrong_binding), patch.object(np, 'load', side_effect=RuntimeError('Unexpected array read')):
        rejects(lambda: reader.Partition('train'))
    checks['verification_wrong_split_binding_rejects_before_arrays'] = True
    with tempfile.TemporaryDirectory(prefix='round05_independent_reader_') as temporary:
        directory = Path(temporary)
        arrays = {'ids': np.arange(6, dtype=np.int64), 'E': np.arange(60, dtype=np.float64).reshape(6,10),
                  'A': np.arange(540, dtype=np.float64).reshape(6,10,3,3)}
        for name in ('E', 'A'):
            arrays[name][[0,2,5]] = np.nan
        with zipfile.ZipFile(directory/'raw_labels.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, values in arrays.items():
                buffer = io.BytesIO(); np.save(buffer, values, allow_pickle=False)
                archive.writestr(name+'.npy', buffer.getvalue())
        part = reader.Partition.__new__(reader.Partition)
        part.indices = np.array([1,3,4], dtype=np.int64)
        part.allowed = np.array([False,True,False,True,True,False])
        part.corpus_size = 6; part.name = 'train'; part.ledger = []; part.verified_sources = {}
        original = reader.convert_block; calls = []
        def convert(raw, dtype, shape, rows):
            assert part.allowed[rows].all(); calls.append(rows.copy())
            return original(raw, dtype, shape, rows)
        with patch.object(reader, 'DATA', directory), patch.dict(reader.PINS, {'raw_labels.npz': sha(directory/'raw_labels.npz')}), patch.object(reader, 'convert_block', side_effect=convert), patch.object(np, 'load', side_effect=RuntimeError('Full array load')), patch.object(np.lib.format, 'read_array', side_effect=RuntimeError('Full member decode')):
            for name, values in arrays.items():
                actual = part.selected('raw_labels.npz', name)
                np.testing.assert_array_equal(actual, values[part.indices])
                assert np.isfinite(actual).all()
        assert len(calls) == 6
        assert all(np.array_equal(np.concatenate(calls[i:i+2]),part.indices) for i in (0,2,4))
        checks['synthetic_id_energy_tensor_strides_decode_selected_rows_only'] = True
    # Synthetic independent reductions reproduce the reviewed original definitions.
    energy = np.arange(70, dtype=np.float64).reshape(7,10)/7
    tensor = np.arange(630, dtype=np.float64).reshape(7,10,3,3)/630
    mean = energy.mean(0)
    energy_direct = np.square(energy-mean).mean()
    energy_scalar = sum((energy[n,s]-mean[s])**2 for n in range(7) for s in range(10))/70
    tensor_direct = np.square(tensor).sum((-2,-1)).mean()
    tensor_scalar = sum(tensor[n,s,i,j]**2 for n in range(7) for s in range(10) for i in range(3) for j in range(3))/70
    assert abs(energy_direct-energy_scalar)<1e-12 and abs(tensor_direct-tensor_scalar)<1e-12
    checks['normalization_scalar_reference_matches_statewise_and_frobenius_definitions'] = True
    output = {'passed': True, 'synthetic_only': True, 'real_data_or_model_read': False,
              'production_statistics_executed': False, 'checks': checks,
              'source_hashes': {name:sha(ROOT/name) for name in ('common.py','partition_data.py','prepare_train_statistics.py','reader_checks.py','run_statistics.py','independent_reader_checks.py')}}
    immutable_json(output, ROOT/'INDEPENDENT_READER_CHECKS.json')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
