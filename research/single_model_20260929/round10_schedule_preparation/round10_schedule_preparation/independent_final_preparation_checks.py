"""Final source/receipt binding only; no scientific imports or binary data reads."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = '7b4bed854c54eb9a1faabcb817b101c4bafe0f80869d7102242f1746b0c2aa2f'
REPORT = 'ec5440d18c2a9f8104293406771fbcc02a01d31aed4b2541353d6e425780e53f'
DECISION = '4d23e6ab826dd2f4df1f4e72939529ff823fd2210a897a0ea4ba0f5dd6494890'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT / 'INDEPENDENT_FINAL_PREPARATION_CHECKS.json'
    assert not output.exists()
    mp = ROOT / 'FROZEN_MANIFEST.json'
    m = read(mp)
    assert sha(mp) == MANIFEST
    source = m['source_hashes']
    assert len(source) == 458
    for name, digest in source.items():
        p = Path(name)
        assert p.is_absolute() and sha(p) == digest, name
        assert p.suffix.lower() in ('.py', '.md', '.json', '.log', '.xml', '.pid', '.exit', '.cfg')
        assert not any(part.startswith('private_') for part in p.parts)
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
    previous_path = ROOT.parent / 'round09_regularization_preparation/FROZEN_MANIFEST.json'
    previous = read(previous_path)
    assert sha(previous_path) == m['inherited_manifest_sha256'] == '52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930'
    assert len(previous['source_hashes']) == 360
    assert all(source.get(p) == h for p, h in previous['source_hashes'].items())
    for field in ('epoch_order_sha256', 'initial_base_tensor_sha256', 'initial_full_tensor_sha256',
                  'initial_transport_tensor_sha256', 'split_manifest_sha256',
                  'independent_split_verification_sha256', 'train_index_sha256', 'raw_archive_opaque_sha256'):
        assert m[field] == previous[field], field
    assert len(m['epoch_order_sha256']) == 60
    cfg = read(ROOT / 'config.json')
    proposal = read(ROOT / 'PROPOSAL_SETTINGS.json')
    assert m['config'] == cfg
    for field in ('seed', 'order_seed', 'epochs', 'batch_size', 'optimizer', 'lr', 'betas', 'eps',
                  'grad_clip', 'scheduler', 'amp', 'tf32', 'dtype', 'num_threads', 'retained_v2_reference_r2',
                  'test_access', 'historical_weights', 'selection', 'train_molecules', 'validation_molecules'):
        assert cfg[field] == previous['config'][field], field
    rates = {'fixed_lr': [.001] * 60, 'step_lr': [.001] * 30 + [.0003] * 30}
    assert list(cfg['arms']) == ['fixed_lr', 'step_lr']
    assert cfg['arms'] == proposal['arms']
    assert m['epoch_learning_rates'] == rates
    for arm in rates:
        assert cfg['arms'][arm] == {'mode': 'original', 'weight_decay': 0., 'epoch_learning_rates': rates[arm]}
    schedule_sha = hashlib.sha256(json.dumps(rates, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert m['schedule_sha256'] == schedule_sha
    assert cfg['batches_per_epoch'] == 1881 and cfg['optimizer_updates'] == 112860
    assert ((cfg['train_molecules'] + 63) // 64) * cfg['epochs'] == 112860
    assert cfg['promotion_delta_r2'] == .003
    assert cfg['promotion_comparators'] == ['contemporaneous selected fixed_lr', 'retained v2 reference']
    assert cfg['retained_v2_reference_r2'] == .44716940136585204
    assert m['model_config'] == read(ROOT / 'model_config.json') == previous['model_config']
    assert sha(ROOT / 'TRAIN_STATISTICS.json') == sha(previous_path.parent / 'TRAIN_STATISTICS.json')
    assert sha(ROOT / 'ROUND10_PREPARATION_DECISION.md') == DECISION
    cpu, gpu = read(ROOT / 'CPU_PREFLIGHT.json'), read(ROOT / 'GPU_PREFLIGHT.json')
    assert cpu['passed'] and gpu['passed']
    assert cpu['optimizer_arithmetic']['toy_optimizer_step_calls'] == m['cpu_toy_optimizer_steps'] == 7
    assert cpu['model_updates'] == 0 and not cpu['real_geometry_or_targets_read']
    assert gpu['executed_optimizer_updates'] == m['actual_discarded_optimizer_updates'] == 6
    assert gpu['unique_train_molecules'] == 128 and gpu['validation_test_numeric_rows'] == 0
    assert cpu['schedule_checks']['schedule_sha256'] == schedule_sha
    receipts = ['CPU_SOURCE_REVIEW.json', 'CPU_PREFLIGHT_REVIEW.json', 'INDEPENDENT_CPU_RESULT_CHECKS.json',
                'INDEPENDENT_GATE_CHECKS.json', 'TECHNICAL_SOURCE_REVIEW.json', 'GPU_PREFLIGHT_REVIEW.json',
                'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json', 'INDEPENDENT_RESOURCE_SELECTOR_CHECKS.json']
    for name in receipts:
        record = read(ROOT / name)
        assert record['passed'], name
        for field in ('source_hashes', 'input_hashes'):
            for p, digest in record.get(field, {}).items():
                resolved = Path(p) if Path(p).is_absolute() else ROOT / p
                assert source.get(str(resolved)) == digest, (name, p)
    assert read(ROOT / 'INDEPENDENT_GATE_CHECKS.json')['checks_count'] == 23
    assert len(read(ROOT / 'TECHNICAL_SOURCE_REVIEW.json')['source_hashes']) == 424
    gpur = read(ROOT / 'GPU_PREFLIGHT_REVIEW.json')
    for field, filename in [('gpu_preflight_sha256', 'GPU_PREFLIGHT.json'),
                            ('technical_review_sha256', 'TECHNICAL_SOURCE_REVIEW.json'),
                            ('independent_result_checks_sha256', 'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json')]:
        assert gpur[field] == sha(ROOT / filename)
    roster = read(ROOT / 'PARAMETER_ROSTER.json')
    assert sha(ROOT / 'PARAMETER_ROSTER.json') == m['parameter_roster_sha256'] == cpu['parameter_roster_sha256'] == gpu['parameter_roster_sha256']
    assert roster['contract'] == read(previous_path.parent / 'PARAMETER_ROSTER.json')['contract']
    assert roster['contract']['included_tensor_count'] == 135 and roster['unwrapped_original_ordered_roster_equal']
    lineage = read(ROOT / 'SOURCE_LINEAGE.json')
    assert not lineage['trained_weights_or_optimizer_states_reused']
    for row in lineage['files'].values():
        assert sha(row['previous_path']) == row['previous_sha256']
        assert sha(row['current_path']) == row['current_sha256']
        assert row['byte_identical'] == (row['previous_sha256'] == row['current_sha256'])
    frozen_proposal = read(ROOT / 'INDEPENDENT_REVIEW_MANIFEST.json')
    assert sha(ROOT / 'INDEPENDENT_REVIEW_MANIFEST.json') == 'fa0d9f7cbb0334710127949f72e8f01ffdd99e49d742092c61ea45d571de2056'
    assert len(frozen_proposal['files']) == 15
    assert all(source.get(p) == h for p, h in frozen_proposal['files'].items())
    preservation_path = ROOT / 'freeze_attempt01_preserved/MANIFEST.json'
    assert sha(preservation_path) == '6b8b365847f7d6da9d5ed4fdefe9f741ea657f0b8f9332b2076db2e425a54bde'
    preservation = read(preservation_path)
    assert len(preservation['files']) == 4
    extra_paths = [preservation_path]
    for row in preservation['files']:
        p = Path(row['snapshot_path'])
        assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
        extra_paths.append(p)
        if Path(row['original_path']).name != 'freeze_preparation.py':
            assert sha(row['original_path']) == row['sha256']
    repair = read(ROOT / 'METADATA_REPAIR_REVIEW.json')
    assert repair['passed'] and repair['one_distinct_metadata_retry_permitted']
    assert sha(ROOT / 'METADATA_REPAIR_REVIEW.json') == 'a1942e3cb5551a144f7eb79007d3fc7628ed189a34cc6e0434a25391df787473'
    assert repair['corrected_freezer_sha256'] == sha(ROOT / 'freeze_preparation.py') == source[str(ROOT / 'freeze_preparation.py')]
    for field in ('source_hashes', 'input_hashes'):
        for p, digest in repair[field].items():
            assert sha(p) == digest
    failed = read(ROOT / 'ops/FREEZE_RECEIPT.json')
    assert failed['passed'] is False and failed['exit_code'] == 1 and failed['frozen_manifest_sha256'] is None
    assert failed['log_sha256'] == sha(ROOT / 'ops/FREEZE_01.log')
    freeze = read(ROOT / 'ops/FREEZE_RECEIPT_02.json')
    assert freeze['passed'] and freeze['exit_code'] == 0
    assert freeze['frozen_manifest_sha256'] == MANIFEST and freeze['source_files'] == 458
    assert freeze['log_sha256'] == sha(ROOT / 'ops/FREEZE_02.log')
    assert freeze['source_sha256'] == sha(ROOT / 'freeze_preparation.py')
    assert freeze['metadata_repair_review_sha256'] == sha(ROOT / 'METADATA_REPAIR_REVIEW.json')
    for rec in (failed, freeze):
        assert rec['optimizer_updates'] == 0 and not rec['target_numeric_decoding']
        assert not rec['model_inference'] and not rec['production_authorized']
        assert rec['environment'] == {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
    assert ast.literal_eval((ROOT / 'ops/FREEZE_02.log').read_text().strip()) == {'files': 458, 'manifest_sha256': MANIFEST}
    assert sha(ROOT / 'PREPARATION_REPORT.md') == REPORT
    assert not (ROOT / 'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists() and not (ROOT / 'runs').exists()
    assert read(ROOT / 'PRODUCTION_AUTHORIZATION_TEMPLATE.json')['authorized'] is False
    extra_paths += [ROOT / name for name in ['METADATA_REPAIR_REVIEW.json', 'independent_metadata_repair.py',
        'INDEPENDENT_METADATA_REPAIR_01.log', 'ops/FREEZE_RECEIPT.json', 'ops/FREEZE_01.log',
        'ops/FREEZE_RECEIPT_02.json', 'ops/FREEZE_02.log', 'PREPARATION_REPORT.md']]
    record = {'passed': True, 'phase': 'final_preparation_metadata_and_source_closure',
        'frozen_manifest_sha256': MANIFEST, 'source_files_verified': 458,
        'inherited_360_sources_unchanged': True, 'executed_424_technical_pins_unchanged_and_in_closure': True,
        'all_60_inherited_order_hashes_unchanged': True, 'accepted_schedule_tables_and_digest_match': True,
        'CPU_seven_toy_calls_zero_model_updates_and_GPU_six_discarded_updates': True,
        'no_validation_or_test_fixture': True, 'roster_contract_unchanged': True,
        'metadata_failure_preserved_and_single_reviewed_retry_passed': True,
        'production_authority_absent_and_no_runs': True, 'production_authorized': False,
        'model_or_data_imports_or_numerical_reexecution_by_checker': False,
        'source_hashes': source,
        'input_hashes': {str(p): sha(p) for p in [mp, Path(__file__)] + extra_paths}}
    with output.open('x') as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': True, 'source_files': 458, 'receipt_sha256': sha(output)}))

if __name__ == '__main__':
    main()
