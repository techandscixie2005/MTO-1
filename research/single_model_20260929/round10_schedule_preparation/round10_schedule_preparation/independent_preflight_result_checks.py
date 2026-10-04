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
    assert set(result['arms'])=={'fixed_lr','step_lr'}
    assert result['resume_tolerances']=={'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,'loss_atol':1e-6,'loss_rtol':1e-5}
    assert terminal['exit_code'] == terminal['registration_returncode'] == 0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert sha(ROOT/'GPU_PREFLIGHT.json') == terminal['preflight_sha256']
    assert sha(ops/'stage.log') == terminal['log_sha256']
    assert result['technical_review_sha256']==launch['review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')=='b686396c5e766aef675b7f744b35d70700d4cc1e265765809f40c210cfd3836c'
    assert sha(ROOT/'GPU_PREFLIGHT.json')=='2bf6eab787acdbd183f4f7df2ad21f33e2c3693b40a4d8e7d686f4b5eab105a1'
    selection=read(ops/'RESOURCE_SELECTION.json')
    assert selection['ordered_allowed']==[1,2,4,6] and selection['selection_under_exclusive_lock']
    assert selection['selected_gpu']==1 and len(selection['observations'])==1
    selected=selection['observations'][0];assert selected['gpu']==1 and selected['eligible']
    assert sha(ops/selected['probe_path'])==selected['probe_sha256']==sha(ops/'GPU_ADMISSION.xml')
    wrapper=read(ROOT/'ops/GPU_WRAPPER_EXECUTION_RECEIPT.json')
    assert wrapper['passed'] and wrapper['exit_code']==0 and wrapper['identity']==launch['wrapper_identity']
    assert wrapper['source_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert wrapper['gpu_preflight_sha256']==sha(ROOT/'GPU_PREFLIGHT.json')
    assert wrapper['log_sha256']==sha(ROOT/'ops/GPU_WRAPPER_01.log')
    assert wrapper['environment']=={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    assert result['source_hashes'] == review['source_hashes']
    for path, digest in result['source_hashes'].items():
        assert sha(path) == digest, path
    assert result['root_preparation_decision_sha256'] == sha(ROOT/'ROUND10_PREPARATION_DECISION.md')
    # R/S describes a changing scheduler state, not process ownership. The
    # barrier deliberately leaves the registered child sleeping before release.
    identity_fields=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
    assert launch['child_identity']==terminal['child_identity']
    assert {k:launch['child_identity'][k] for k in identity_fields}=={
        k:registration['identity'][k] for k in identity_fields}
    assert registration['id'] == 'round10_technical_preflight'
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
    assert [x['update'] for x in updates] == list(range(1, 7))
    assert [x['arm'] for x in updates] == ['fixed_lr']*3 + ['step_lr']*3
    assert [x['schedule_label'] for x in updates]==[30,31,31]*2
    assert [x['actual_adam_step'] for x in updates]==[1,2,2]*2
    assert [x['learning_rate'] for x in updates]==[.001,.001,.001,.001,.0003,.0003]
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
    checkpoint_hashes = {}
    for arm, row in result['arms'].items():
        assert arm in ('fixed_lr', 'step_lr') and row['updates'] == 3
        assert row['parameter_roster_sha256']==roster_sha and row['optimizer_parameter_tensor_count']==135
        assert row['weight_decay']==0. and row['frozen_parameters_and_buffers_unchanged']
        assert row['fixture_order_sha256'] == result['train_indices_sha256']
        assert row['schedule_labels']==[30,31,31] and row['actual_adam_counters']==[1,2,2]
        expected_lrs=[.001]*3 if arm=='fixed_lr' else [.001,.0003,.0003]
        assert row['applied_learning_rates']==expected_lrs
        assert row['moments_unchanged_by_boundary_assignment'] and row['fixture_is_not30_completed_epochs']
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
            assert gradients['schedule_label']==[30,31,31][j] and gradients['actual_adam_step_after']==[1,2,2][j]
            assert gradients['learning_rate']==expected_lrs[j]
            assert gradients['unplanned_setting_change'] is False
            assert gradients['direct_trace_A_gradient_l2'] > 0
            assert gradients['direct_trace_E_gradient_l2'] >= 0
            assert gradients['direct_trace_E_gradient_l2']==0
            assert all(v>0 and math.isfinite(v) for v in gradients['term_parameter_gradient_l2'].values())
            branches=gradients['preclip_branch_gradient_l2']
            assert branches['base_gradient_l2']>0
            assert branches['gate_gradient_l2']==0
            reg=gradients['regularization_diagnostics'];assert reg['parameters_with_grad']==135 and reg['parameters_grad_none']==0
            assert all(math.isfinite(v) for v in reg.values())
            decay=0.
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
        ROOT/'CPU_PREFLIGHT_REVIEW.json', ROOT/'ops/GPU_WRAPPER_01.log', ROOT/'ops/GPU_WRAPPER_EXECUTION_RECEIPT.json',
        ROOT/'INDEPENDENT_TECHNICAL_SOURCE_BINDING_01.log', ops/'RESOURCE_SELECTION.json', ops/'GPU1_PROBE.xml',
        ROOT.parent/'round05_scratch_preparation/GPU_PREFLIGHT.json',
        ops/'COMPLETE.json', ops/'LAUNCH_RECEIPT.json', ops/'REGISTRATION_TOOL.log', ops/'GPU_ADMISSION.xml', ops/'stage.log', Path(__file__)]}
    output.write_text(json.dumps({'passed': True, 'phase': 'completed_round10_six_update_and_schedule_receipt_review',
        'source_bindings_verified': len(result['source_hashes']), 'input_hashes': bindings,
        'private_checkpoint_opaque_hashes_only': checkpoint_hashes,
        'exact_six_updates_and_first128_TRAIN_verified': True,
        'same_fixture_as_round05': True, 'owned_registration_admission_and_exit0_verified': True,
        'registered_entry_before_updates_and_terminal_child_absence_verified': True,
        'fixed_replay_tolerances_and_exact_RNG_order_passed': True,
        'no_data_tensor_decode_or_model_inference_by_reviewer': True,
        'one_pass_first_eligible_GPU1_admission_verified':True,
        'schedule_boundary_and_actual_counters_verified':True,
        'original_scientific_source_and_six_update_budget_unchanged':True,
        'production_fit_authorized': False}, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'source_bindings': len(result['source_hashes'])}))


if __name__ == '__main__':
    main()
