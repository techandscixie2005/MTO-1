"""Final metadata/source verification; no model imports or numeric target reads."""
import ast
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_MANIFEST = '66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    manifest_path = ROOT / 'FROZEN_MANIFEST.json'
    assert sha(manifest_path) == EXPECTED_MANIFEST
    manifest = read(manifest_path)
    assert len(manifest['source_hashes']) == 82
    for path, digest in manifest['source_hashes'].items():
        assert sha(path) == digest, path
        assert Path(path).suffix not in ('.pt', '.pth', '.npy', '.npz')
    split_path = ROOT.parent / 'dataset_audit_20260930/SPLIT_MANIFEST.json'
    split = read(split_path)
    assert sha(split_path) == manifest['split_manifest_sha256']
    assert split['counts'] == {'train':120355, 'val':6686, 'test':6686}
    spec = split['arrays']['train_indices.npy']
    assert sha(spec['path']) == spec['sha256']
    # Index metadata only. Do not open any raw NPZ, geometry, label or prediction member.
    rows = np.load(spec['path'], allow_pickle=False)
    assert rows.dtype == np.dtype('int64') and rows.shape == (120355,)
    assert np.all(rows[1:] > rows[:-1])
    row_hash = hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
    assert row_hash == spec['content_sha256'] == manifest['train_index_sha256']
    generator = np.random.Generator(np.random.PCG64(11))
    orders = [hashlib.sha256(rows[generator.permutation(len(rows))].astype('<i8').tobytes()).hexdigest() for _ in range(60)]
    assert orders == manifest['epoch_order_sha256']
    cfg = manifest['config']
    assert cfg == read(ROOT / 'config.json')
    assert cfg['epochs'] == 60 and cfg['batch_size'] == 64 and cfg['lr'] == .001
    assert cfg['seed'] == cfg['order_seed'] == 11
    assert not cfg['test_access'] and not cfg['historical_weights']
    cpu, gpu = read(ROOT/'CPU_PREFLIGHT.json'), read(ROOT/'GPU_PREFLIGHT.json')
    assert cpu['passed'] and gpu['passed'] and gpu['executed_optimizer_updates'] == 12
    for key in ('initial_base_tensor_sha256', 'initial_full_tensor_sha256'):
        assert cpu[key] == manifest[key]
    io = read(ROOT/'IO_CONTINUITY.json')
    ioreview = read(ROOT/'IO_CONTINUITY_REVIEW.json')
    assert ioreview['passed'] and ioreview['mapping_sha256'] == sha(ROOT/'IO_CONTINUITY.json')
    assert sha(io['executed_snapshot_path']) == io['executed_sha256']
    assert sha(io['source_path']) == io['final_sha256']
    node = next(n for n in ast.parse((ROOT/'freeze_preparation.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name == 'verify_receipt_source')
    # The original reviewer hash used Python3.12, which adds empty type_params
    # to FunctionDef. Match that representation on the server's Python3.10.
    if 'type_params' not in node._fields:
        node._fields = node._fields + ('type_params',)
        node.type_params = []
    assert hashlib.sha256(ast.dump(node).encode()).hexdigest() == ioreview['freezer_exception_function_ast_sha256']
    checked = 0
    for receipt_name in ('CPU_PREFLIGHT.json','GPU_PREFLIGHT.json','READER_CHECKS.json',
                         'TRAIN_STATISTICS_AUDIT.json','INDEPENDENT_GATE_CHECKS.json',
                         'TECHNICAL_SOURCE_REVIEW.json','TECHNICAL_SOURCE_REVIEW_RETRY01.json'):
        receipt = read(ROOT/receipt_name)
        assert receipt['passed']
        for name, digest in receipt.get('source_hashes', {}).items():
            path = Path(name)
            if not path.is_absolute(): path = ROOT/path
            if sha(path) != digest:
                assert str(path) == io['source_path'] and digest == io['executed_sha256']
                assert sha(path) == io['final_sha256']
            checked += 1
    for name in ('TRAIN_STATISTICS_REVIEW.json','GPU_PREFLIGHT_REVIEW.json','IO_CONTINUITY_REVIEW.json'):
        assert read(ROOT/name)['passed']
    assert read(ROOT/'PRODUCTION_AUTHORIZATION_TEMPLATE.json')['authorized'] is False
    assert not (ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    assert not (ROOT/'runs').exists(), 'No production run directory is expected before authorization'
    result = {
        'passed':True, 'frozen_manifest_sha256':EXPECTED_MANIFEST,
        'current_frozen_files_verified':82, 'receipt_source_entries_verified':checked,
        'all_60_train_order_hashes_independently_recomputed':True,
        'only_private_array_opened':'TRAIN index metadata; no numeric target/geometry/prediction arrays',
        'numeric_target_or_prediction_values_read':False, 'model_inference_or_optimizer_updates':False,
        'production_authority_or_run_artifacts_present':False,
        'reviewer_script_sha256':sha(__file__),
        'io_continuity_is_the_only_receipt_source_exception':True,
    }
    out = ROOT/'FINAL_BINDINGS_CHECKS.json'
    assert not out.exists(), 'Never overwrite a completed check'
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__': main()
