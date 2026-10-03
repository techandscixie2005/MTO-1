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
    assert set(result['arms'])=={'original','local','neighbor'}
    assert result['resume_tolerances']=={'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,'loss_atol':1e-6,'loss_rtol':1e-5}
    assert terminal['exit_code'] == terminal['registration_returncode'] == 0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert sha(ROOT/'GPU_PREFLIGHT.json') == terminal['preflight_sha256']
    assert sha(ops/'stage.log') == terminal['log_sha256']
    assert result['technical_review_sha256'] == launch['review_sha256'] == sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert result['source_hashes'] == review['source_hashes']
    for path, digest in result['source_hashes'].items():
        assert sha(path) == digest, path
    assert result['root_preparation_decision_sha256'] == sha(ROOT/'ROUND08_PREPARATION_DECISION.md')
    assert launch['child_identity'] == terminal['child_identity'] == registration['identity']
    assert registration['id'] == 'round08_technical_preflight'
    uuid = 'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'
    assert launch['gpu'] == 1
    assert launch['registration_barrier_released'] and launch['registration_returncode']==0
    assert launch['command'][1:]==[str(ROOT/'registered_entry.py'),str(ROOT/'gpu_preflight.py')]
    assert launch['child_pid']==launch['child_identity']['pid']
    assert not Path('/proc/%d' % launch['child_pid']).exists(), 'Owned numerical child still exists'
    lines=(ops/'stage.log').read_text().splitlines()
    barrier_lines=[i for i,line in enumerate(lines) if line.startswith('{"registration_barrier_passed":')]
    update_lines=[i for i,line in enumerate(lines) if line.startswith('{"update":')]
    assert len(barrier_lines)==1 and update_lines and barrier_lines[0]<min(update_lines)
    assert json.loads(lines[barrier_lines[0]])['registration_barrier_passed']=={
        'registered':True,'pid':launch['child_pid'], 'binding_sha256':sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')}

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
    assert [x['update'] for x in updates] == list(range(1, 10))
    assert [x['arm'] for x in updates] == ['original']*3 + ['local']*3 + ['neighbor']*3
    assert read(ROOT/'GPU_UPDATE_PROGRESS.json')['executed_optimizer_updates'] == 9
    assert result['executed_optimizer_updates'] == 9
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
        assert arm in ('original', 'local', 'neighbor') and row['updates'] == 3
        assert row['fixture_order_sha256'] == result['train_indices_sha256']
        assert row['rng_and_order_exact_before_and_after_replay']
        for key in ('model_replay_max_abs','optimizer_replay_max_abs','loss_replay_abs'):
            assert math.isfinite(row[key]) and row[key] >= 0
        # The reviewed compare/allclose calls enforce combined absolute+relative tolerances.
        # Aggregate maxima alone cannot reconstruct a relative-tolerance residual.
        assert row['geometry_checkpoint_load_forward_numerical_pass']
        for name in ('initial_output_errors', 'geometry_checkpoint_output_errors'):
            assert all(math.isfinite(row[name][k]) and row[name][k] >= 0 for k in ('E_max_abs','A_max_abs','native_f_max_abs'))
        assert row['clipped'] == [v > 5 for v in row['preclip_norms']]
        assert len(row['update_losses']) == len(row['gradient_audits']) == 3
        for j, (loss, gradients) in enumerate(zip(row['update_losses'],row['gradient_audits'])):
            assert math.isclose(loss['total'], loss['energy']+loss['trace'], abs_tol=1e-6, rel_tol=1e-5)
            assert loss['base'] == loss['total']
            assert gradients['preupdate_losses'] == loss
            assert gradients['coefficient_or_LR_adjusted'] is False
            assert gradients['direct_trace_A_gradient_l2'] > 0
            assert gradients['direct_trace_E_gradient_l2'] >= 0
            assert gradients['direct_trace_E_gradient_l2']==0
            assert all(v>0 and math.isfinite(v) for v in gradients['term_parameter_gradient_l2'].values())
            branches=gradients['preclip_branch_gradient_l2']
            assert branches['base_gradient_l2']>0
            if arm=='original': assert branches['gate_gradient_l2']==0
            else: assert branches['gate_gradient_l2']>0
            assert all(math.isfinite(v) for v in gradients['transport_diagnostics'].values())
        for filename, field in ((arm+'_after_update1.pt', 'checkpoint_sha256'), (arm+'_geometry_fixture.pt', 'geometry_checkpoint_sha256')):
            path = ROOT/'private_preflight'/filename
            assert sha(path) == row[field]
            checkpoint_hashes[str(path)] = row[field]
    assert not (ops/'FAILED.json').exists()
    bindings = {str(p): sha(p) for p in [ROOT/'GPU_PREFLIGHT.json', ROOT/'TECHNICAL_SOURCE_REVIEW.json',
        ROOT/'CPU_PREFLIGHT_REVIEW.json', ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json', ROOT/'GPU_UPDATE_PROGRESS.json',
        ROOT.parent/'round05_scratch_preparation/GPU_PREFLIGHT.json',
        ops/'COMPLETE.json', ops/'LAUNCH_RECEIPT.json', ops/'REGISTRATION_TOOL.log', ops/'GPU_ADMISSION.xml', ops/'stage.log', Path(__file__)]}
    output.write_text(json.dumps({'passed': True, 'phase': 'completed_nine_update_receipt_review',
        'source_bindings_verified': len(result['source_hashes']), 'input_hashes': bindings,
        'private_checkpoint_opaque_hashes_only': checkpoint_hashes,
        'exact_nine_updates_and_first128_TRAIN_verified': True,
        'same_fixture_as_round05': True, 'owned_registration_admission_and_exit0_verified': True,
        'registered_entry_before_updates_and_terminal_child_absence_verified': True,
        'fixed_replay_tolerances_and_exact_RNG_order_passed': True,
        'no_data_tensor_decode_or_model_inference_by_reviewer': True,
        'production_fit_authorized': False}, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'source_bindings': len(result['source_hashes'])}))


if __name__ == '__main__':
    main()
