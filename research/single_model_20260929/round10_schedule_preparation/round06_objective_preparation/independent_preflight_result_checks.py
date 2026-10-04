"""Read completed aggregate receipts and opaque hashes; never import a model."""
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT/'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json'
    assert not output.exists(), 'Preserve completed checks'
    review = read(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    result = read(ROOT/'GPU_PREFLIGHT.json')
    ops = ROOT/'ops/gpu_preflight_attempt'
    terminal = read(ops/'COMPLETE.json')
    launch = read(ops/'LAUNCH_RECEIPT.json')
    registration = read(ops/'REGISTRATION_TOOL.log')
    assert result['passed'] and terminal['passed']
    assert terminal['exit_code'] == terminal['registration_returncode'] == 0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert sha(ROOT/'GPU_PREFLIGHT.json') == terminal['preflight_sha256']
    assert sha(ops/'stage.log') == terminal['log_sha256']
    assert result['technical_review_sha256'] == launch['review_sha256'] == sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert result['source_hashes'] == review['source_hashes']
    for path, digest in result['source_hashes'].items():
        assert sha(path) == digest, path
    assert result['root_preparation_decision_sha256'] == sha(ROOT/'ROUND06_PREPARATION_DECISION.md')
    assert launch['child_identity'] == terminal['child_identity'] == registration['identity']
    assert registration['id'] == 'round06_technical_preflight'
    uuid = 'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'
    assert launch['gpu'] == 1
    assert result['device_uuid_environment'] == launch['gpu_uuid'] == registration['gpu_uuid'] == uuid
    assert launch['environment'] == dict(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    xml = ET.parse(ops/'GPU_ADMISSION.xml').getroot().find('gpu')
    assert xml.findtext('uuid') == uuid
    assert xml.findtext('ecc_mode/current_ecc') == 'Enabled'
    assert xml.findtext('gpu_recovery_action') == 'None'
    assert int(xml.findtext('fb_memory_usage/used').split()[0]) < 1000
    assert all('C' not in node.findtext('type', '') for node in xml.findall('processes/process_info'))
    for period in ('volatile', 'aggregate'):
        for field in ('sram_uncorrectable_parity', 'sram_uncorrectable_secded', 'dram_uncorrectable'):
            assert xml.findtext('ecc_errors/'+period+'/'+field) == '0'
    for field in ('remapped_row_pending', 'remapped_row_failure'):
        assert xml.findtext('remapped_rows/'+field) == 'No'
    for field in ('channel_repair_pending', 'tpc_repair_pending'):
        assert xml.findtext('ecc_errors/'+field) in ('No', 'N/A')
    updates = []
    for line in (ops/'stage.log').read_text().splitlines():
        if line.startswith('{"update":'):
            updates.append(json.loads(line))
    assert [x['update'] for x in updates] == list(range(1, 7))
    assert [x['arm'] for x in updates] == ['trace_control']*3 + ['raw_f']*3
    assert read(ROOT/'GPU_UPDATE_PROGRESS.json')['executed_optimizer_updates'] == 6
    assert result['executed_optimizer_updates'] == 6
    assert result['unique_train_molecules'] == 128 and result['validation_test_numeric_rows'] == 0
    prior = read(ROOT.parent/'round05_scratch_preparation/GPU_PREFLIGHT.json')
    for field in ('train_indices_sha256', 'train_ids_sha256'):
        assert result[field] == prior[field]
    assert len(result['actual_decode_ledger']) == 12
    for entry in result['actual_decode_ledger']:
        assert entry['partition'] == 'train' and entry['numeric_rows'] == 128
        assert entry['numeric_global_indices_sha256'] == result['train_indices_sha256']
        assert entry['shape'][0] == 128
    checkpoint_hashes = {}
    for arm, row in result['arms'].items():
        assert arm in ('trace_control', 'raw_f') and row['updates'] == 3
        assert row['fixture_order_sha256'] == result['train_indices_sha256']
        assert row['rng_and_order_exact_before_and_after_replay']
        assert row['model_replay_max_abs'] <= 2e-6
        assert row['optimizer_replay_max_abs'] <= 2e-6
        assert row['loss_replay_abs'] <= 1e-6
        assert row['geometry_checkpoint_load_forward_numerical_pass']
        for name in ('initial_output_errors', 'geometry_checkpoint_output_errors'):
            assert row[name]['E_max_abs'] <= 2e-6 and row[name]['A_max_abs'] <= 2e-6
        assert row['clipped'] == [v > 5 for v in row['preclip_norms']]
        for loss in row['update_losses']:
            intensity = loss['trace'] if arm == 'trace_control' else loss['raw_f']
            assert math.isclose(loss['total'], loss['energy']+intensity, abs_tol=1e-6, rel_tol=1e-5)
            assert math.isclose(loss['base'], loss['energy']+loss['trace'], abs_tol=1e-6, rel_tol=1e-5)
        gradients = row['gradient_audit']
        assert gradients['coefficient_or_LR_adjusted'] is False
        assert gradients['direct_output_gradients']['trace']['E_gradient_l2'] == 0
        assert all(gradients['direct_output_gradients']['raw_f'][k] > 0 for k in ('E_gradient_l2', 'A_gradient_l2'))
        assert gradients['objective_energy_head_gradient_l2'] > 0 and gradients['objective_PSD_heads_gradient_l2'] > 0
        for filename, field in ((arm+'_after_update1.pt', 'checkpoint_sha256'), (arm+'_geometry_fixture.pt', 'geometry_checkpoint_sha256')):
            path = ROOT/'private_preflight'/filename
            assert sha(path) == row[field]
            checkpoint_hashes[str(path)] = row[field]
    assert not (ops/'FAILED.json').exists()
    bindings = {str(p): sha(p) for p in [ROOT/'GPU_PREFLIGHT.json', ROOT/'TECHNICAL_SOURCE_REVIEW.json',
        ROOT/'CPU_PREFLIGHT_REVIEW.json', ROOT/'GPU_UPDATE_PROGRESS.json', ROOT/'ops/GPU_PRELAUNCH_GUARD_FAILURE.md',
        ops/'COMPLETE.json', ops/'LAUNCH_RECEIPT.json', ops/'REGISTRATION_TOOL.log', ops/'GPU_ADMISSION.xml', ops/'stage.log', Path(__file__)]}
    output.write_text(json.dumps({'passed': True, 'phase': 'completed_six_update_receipt_review',
        'source_bindings_verified': len(result['source_hashes']), 'input_hashes': bindings,
        'private_checkpoint_opaque_hashes_only': checkpoint_hashes,
        'exact_six_updates_and_first128_TRAIN_verified': True,
        'same_fixture_as_round05': True, 'owned_registration_admission_and_exit0_verified': True,
        'fixed_replay_tolerances_and_exact_RNG_order_passed': True,
        'no_data_tensor_decode_or_model_inference_by_reviewer': True,
        'prelaunch_missing_mirror_preserved_zero_updates': True, 'production_fit_authorized': False}, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'source_bindings': len(result['source_hashes'])}))


if __name__ == '__main__':
    main()
