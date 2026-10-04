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
    retry = read(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    ops = ROOT/'ops/gpu_preflight_attempt_retry01'
    terminal = read(ops/'COMPLETE.json')
    launch = read(ops/'LAUNCH_RECEIPT.json')
    registration = read(ops/'REGISTRATION_TOOL.log')
    assert result['passed'] and terminal['passed']
    assert set(result['arms'])=={'zero_decay','coupled_l2'}
    assert result['resume_tolerances']=={'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,'loss_atol':1e-6,'loss_rtol':1e-5}
    assert terminal['exit_code'] == terminal['registration_returncode'] == 0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert sha(ROOT/'GPU_PREFLIGHT.json') == terminal['preflight_sha256']
    assert sha(ops/'stage.log') == terminal['log_sha256']
    assert result['technical_review_sha256'] == launch['original_technical_review_sha256'] == terminal['original_technical_review_sha256'] == sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert launch['review_sha256'] == terminal['resource_retry_review_sha256'] == sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert terminal['resource_retry_decision_unchanged']
    assert retry['root_resource_retry_decision_sha256'] == launch['root_resource_retry_decision_sha256'] == sha(ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md')
    for path,digest in retry['source_hashes'].items():assert sha(path)==digest,path
    assert result['source_hashes'] == review['source_hashes']
    for path, digest in result['source_hashes'].items():
        assert sha(path) == digest, path
    assert result['root_preparation_decision_sha256'] == sha(ROOT/'ROUND09_PREPARATION_DECISION.md')
    # R/S describes a changing scheduler state, not process ownership. The
    # barrier deliberately leaves the registered child sleeping before release.
    identity_fields=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
    assert launch['child_identity']==terminal['child_identity']
    assert {k:launch['child_identity'][k] for k in identity_fields}=={
        k:registration['identity'][k] for k in identity_fields}
    assert registration['id'] == 'round09_technical_preflight_retry01'
    uuid = 'GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184'
    assert launch['gpu'] == 4
    assert launch['registration_barrier_released'] and launch['registration_returncode']==0
    assert launch['command'][1:]==[str(ROOT/'resource_retry_entry.py'),str(ROOT/'gpu_preflight.py')]
    assert launch['child_pid']==launch['child_identity']['pid']
    assert not Path('/proc/%d' % launch['child_pid']).exists(), 'Owned numerical child still exists'
    lines=(ops/'stage.log').read_text().splitlines()
    barrier_lines=[i for i,line in enumerate(lines) if line.startswith('{"registration_barrier_passed":')]
    update_lines=[i for i,line in enumerate(lines) if line.startswith('{"update":')]
    assert len(barrier_lines)==1 and update_lines and barrier_lines[0]<min(update_lines)
    assert json.loads(lines[barrier_lines[0]])['registration_barrier_passed']=={
        'registered':True,'pid':launch['child_pid'], 'binding_sha256':sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')}

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
    assert [x['arm'] for x in updates] == ['zero_decay']*3 + ['coupled_l2']*3
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
    roster_sha=sha(ROOT/'PARAMETER_ROSTER.json');assert result['parameter_roster_sha256']==read(ROOT/'CPU_PREFLIGHT.json')['parameter_roster_sha256']==roster_sha
    assert not list((ROOT/'ops/gpu_preflight_attempt').iterdir())
    old=read(ROOT/'admission01_preserved/MANIFEST.json')
    for entry in old['files']:
        for key in ('original_path','snapshot_path'):assert sha(entry[key])==entry['sha256']
    checkpoint_hashes = {}
    for arm, row in result['arms'].items():
        assert arm in ('zero_decay', 'coupled_l2') and row['updates'] == 3
        assert row['parameter_roster_sha256']==roster_sha and row['optimizer_parameter_tensor_count']==135
        assert row['weight_decay']==(0. if arm=='zero_decay' else 1e-4) and row['frozen_parameters_and_buffers_unchanged']
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
            assert branches['gate_gradient_l2']==0
            reg=gradients['regularization_diagnostics'];assert reg['parameters_with_grad']==135 and reg['parameters_grad_none']==0
            assert all(math.isfinite(v) for v in reg.values())
            decay=0. if arm=='zero_decay' else 1e-4
            assert math.isclose(reg['coupled_term_l2'],decay*reg['trainable_parameter_l2'],rel_tol=1e-12,abs_tol=1e-14)
            assert math.isclose(reg['regularized_decay_task_ratio'],reg['coupled_term_l2']/max(reg['postclip_task_gradient_l2'],1e-12),rel_tol=1e-12,abs_tol=1e-14)
            assert reg['postclip_task_gradient_l2']<=5.00001
            assert abs(reg['postclip_task_gradient_l2']-reg['coupled_term_l2'])-1e-10<=reg['effective_gradient_l2']<=reg['postclip_task_gradient_l2']+reg['coupled_term_l2']+1e-10
            assert all(math.isfinite(v) for v in gradients['transport_diagnostics'].values())
        for filename, field in ((arm+'_after_update1.pt', 'checkpoint_sha256'), (arm+'_geometry_fixture.pt', 'geometry_checkpoint_sha256')):
            path = ROOT/'private_preflight'/filename
            assert sha(path) == row[field]
            checkpoint_hashes[str(path)] = row[field]
    assert not (ops/'FAILED.json').exists()
    bindings = {str(p): sha(p) for p in [ROOT/'GPU_PREFLIGHT.json', ROOT/'TECHNICAL_SOURCE_REVIEW.json',
        ROOT/'CPU_PREFLIGHT_REVIEW.json', ROOT/'CPU_PREFLIGHT.json', ROOT/'PARAMETER_ROSTER.json', ROOT/'GPU_UPDATE_PROGRESS.json',
        ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json', ROOT/'RESOURCE_RETRY_SELECTION.json', ROOT/'RESOURCE_RETRY_CONTINUITY.json',
        ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md', ROOT/'INDEPENDENT_ADMISSION_FAILURE_REVIEW.json', ROOT/'admission01_preserved/MANIFEST.json',
        ROOT.parent/'round05_scratch_preparation/GPU_PREFLIGHT.json',
        ops/'COMPLETE.json', ops/'LAUNCH_RECEIPT.json', ops/'REGISTRATION_TOOL.log', ops/'GPU_ADMISSION.xml', ops/'stage.log', Path(__file__)]}
    output.write_text(json.dumps({'passed': True, 'phase': 'completed_six_update_receipt_and_resource_retry_review',
        'source_bindings_verified': len(result['source_hashes']), 'input_hashes': bindings,
        'private_checkpoint_opaque_hashes_only': checkpoint_hashes,
        'exact_six_updates_and_first128_TRAIN_verified': True,
        'same_fixture_as_round05': True, 'owned_registration_admission_and_exit0_verified': True,
        'registered_entry_before_updates_and_terminal_child_absence_verified': True,
        'fixed_replay_tolerances_and_exact_RNG_order_passed': True,
        'no_data_tensor_decode_or_model_inference_by_reviewer': True,
        'preserved_original_admission_evidence_verified':True,
        'resource_retry_binding_and_GPU4_admission_verified':True,
        'original_scientific_source_and_six_update_budget_unchanged':True,
        'production_fit_authorized': False}, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'source_bindings': len(result['source_hashes'])}))


if __name__ == '__main__':
    main()
