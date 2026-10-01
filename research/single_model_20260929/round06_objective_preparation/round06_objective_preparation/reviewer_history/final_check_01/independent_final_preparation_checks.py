"""Final preparation bindings only: no scientific imports or numeric data reads."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = '4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT/'INDEPENDENT_FINAL_PREPARATION_CHECKS.json'
    assert not output.exists(), 'Do not repeat a completed review'
    manifest_path = ROOT/'FROZEN_MANIFEST.json'
    assert sha(manifest_path) == EXPECTED
    manifest = read(manifest_path)
    source = manifest['source_hashes']
    assert len(source) == 134
    for name, digest in source.items():
        path = Path(name)
        assert path.is_absolute() and sha(path) == digest, name
        assert path.suffix.lower() in ('.py', '.md', '.json', '.log', '.xml')
        assert not any(p.startswith('private_') for p in path.parts)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'))
    previous = read(ROOT.parent/'round05_scratch_preparation/FROZEN_MANIFEST.json')
    assert len(previous['source_hashes']) == 82
    assert all(source.get(name) == digest for name, digest in previous['source_hashes'].items())
    for name in ('epoch_order_sha256', 'initial_base_tensor_sha256', 'initial_full_tensor_sha256',
                 'split_manifest_sha256', 'independent_split_verification_sha256', 'train_index_sha256'):
        assert manifest[name] == previous[name], name
    assert len(manifest['epoch_order_sha256']) == 60
    assert manifest['actual_discarded_optimizer_updates'] == 6
    assert manifest['preparation_only_no_production_authorization']
    cfg = read(ROOT/'config.json')
    proposal = read(ROOT/'PROPOSED_SETTINGS.json')
    assert cfg == manifest['config']
    for name in ('seed', 'order_seed', 'epochs', 'batch_size', 'optimizer', 'lr', 'betas', 'eps',
                 'weight_decay', 'grad_clip', 'scheduler', 'amp', 'tf32', 'selection',
                 'retained_v2_reference_r2', 'test_access', 'historical_weights'):
        assert cfg[name] == proposal[name], name
    assert cfg['num_threads'] == proposal['threads'] == 2
    assert cfg['promotion_delta_r2'] == proposal['candidate_delta_r2_threshold'] == .003
    assert cfg['arms'] == {'trace_control': {'adapter': False, 'objective': 'trace'},
                           'raw_f': {'adapter': False, 'objective': 'raw_f'}}
    assert list(cfg['arms']) == ['trace_control', 'raw_f']
    assert cfg['train_molecules'] == 120355 and cfg['validation_molecules'] == 6686
    assert ((cfg['train_molecules']+63)//64)*cfg['epochs'] == proposal['updates_per_arm'] == 112860
    stats = read(ROOT/'TRAIN_STATISTICS.json')
    for name, proposed in [('sE2', 'sE2'), ('sA2', 'sA2'), ('f_variance', 'train_raw_f_population_variance')]:
        assert stats[name] == proposal[proposed]
    assert sha(ROOT/'TRAIN_STATISTICS.json') == proposal['train_statistics_sha256']
    assert manifest['model_config'] == read(ROOT/'model_config.json')
    gpu = read(ROOT/'GPU_PREFLIGHT.json')
    cpu = read(ROOT/'CPU_PREFLIGHT.json')
    assert gpu['executed_optimizer_updates'] == 6 and cpu['model_updates'] == 0
    assert gpu['validation_test_numeric_rows'] == 0 and gpu['unique_train_molecules'] == 128
    assert all(source.get(name) == digest for name, digest in gpu['source_hashes'].items())
    for name in ('CPU_SOURCE_REVIEW.json', 'CPU_PREFLIGHT_REVIEW.json', 'INDEPENDENT_GATE_CHECKS.json',
                 'TECHNICAL_SOURCE_REVIEW.json', 'GPU_PREFLIGHT_REVIEW.json', 'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json'):
        record = read(ROOT/name)
        assert record['passed']
        for field in ('source_hashes', 'input_hashes'):
            assert all(source.get(path) == digest for path, digest in record.get(field, {}).items()), name
    assert read(ROOT/'INDEPENDENT_GATE_CHECKS.json')['checks_count'] == 22
    gpu_review = read(ROOT/'GPU_PREFLIGHT_REVIEW.json')
    for key, filename in [('gpu_preflight_sha256', 'GPU_PREFLIGHT.json'),
                          ('technical_review_sha256', 'TECHNICAL_SOURCE_REVIEW.json'),
                          ('independent_result_checks_sha256', 'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json')]:
        assert gpu_review[key] == sha(ROOT/filename)
    freeze = read(ROOT/'ops/FREEZE_RECEIPT.json')
    assert freeze['passed'] and freeze['exit_code'] == 0
    assert freeze['frozen_manifest_sha256'] == EXPECTED and freeze['source_files'] == 134
    assert freeze['optimizer_updates'] == 0 and not freeze['target_numeric_decoding']
    assert not freeze['model_inference'] and not freeze['production_authorized']
    lineage = read(ROOT/'SOURCE_LINEAGE.json')
    assert lineage['trained_weights_or_optimizer_states_reused'] is False
    for row in lineage['files'].values():
        assert sha(row['previous_path']) == row['previous_sha256']
        assert sha(row['current_path']) == row['current_sha256']
        assert row['byte_identical'] == (row['previous_sha256'] == row['current_sha256'])
    assert not (ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    assert not (ROOT/'runs').exists()
    assert read(ROOT/'PRODUCTION_AUTHORIZATION_TEMPLATE.json')['authorized'] is False
    report = ROOT/'PREPARATION_REPORT.md'
    record = {'passed': True, 'phase': 'final_preparation_metadata_and_source_closure',
              'frozen_manifest_sha256': EXPECTED, 'source_files_verified': len(source),
              'all_82_inherited_sources_unchanged': True, 'all_executed_technical_sources_in_closure': True,
              'all_60_inherited_order_hashes_unchanged': True, 'accepted_scientific_settings_match': True,
              'six_discarded_updates_only_and_no_validation_test_fixture': True,
              'production_authorization_absent_and_no_runs_directory': True,
              'no_model_inference_or_numeric_data_decode_by_checker': True,
              'input_hashes': {str(p): sha(p) for p in [manifest_path, ROOT/'ops/FREEZE_RECEIPT.json', report, Path(__file__)]},
              'source_hashes': source}
    output.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'source_files': len(source), 'receipt_sha256': sha(output)}))


if __name__ == '__main__':
    main()
