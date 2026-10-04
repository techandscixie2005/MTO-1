"""Independent text/aggregate and opaque-byte verification; never decodes arrays."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = '52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930'
AUTH = 'fc16c206e829c345098fcd853febcd2f83c88a56c3ea92f5506c0edd67678765'
MONITOR = '5121f996ae570e2c0525d9e57762af99af19e628f0a8711b059f68aa8c2e0c7d'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), (a, b)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT/'completion/INDEPENDENT_TERMINAL_METADATA.json'
    assert not output.exists(), 'Completed review must not repeat'
    inputs = {}

    def read(p):
        p = Path(p)
        inputs[str(p)] = sha(p)
        return json.loads(p.read_bytes())

    manifest = read(ROOT/'FROZEN_MANIFEST.json')
    assert inputs[str(ROOT/'FROZEN_MANIFEST.json')] == MANIFEST
    assert len(manifest['source_hashes']) == 360
    for p, h in manifest['source_hashes'].items():
        assert sha(p) == h, p
    auth = read(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')
    assert inputs[str(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')] == AUTH
    assert auth['authorized'] and auth['scope'] == 'round09_two_arm_60epoch_fit'
    assert auth['arms'] == ['zero_decay', 'coupled_l2'] and auth['epochs'] == 60
    assert not auth['test_access'] and not auth['historical_weights']
    assert auth['frozen_manifest_sha256'] == MANIFEST
    assert sha(ROOT/'INDEPENDENT_PREPARATION_REVIEW.json') == auth['independent_review_sha256']
    pub = read(auth['publication_receipt'])
    assert inputs[auth['publication_receipt']] == auth['publication_receipt_sha256']
    assert pub['remote_verified'] and pub['download_before_stage_before_commit_push']
    assert pub['frozen_manifest_sha256'] == MANIFEST
    splitpath = ROOT.parent/'dataset_audit_20260930/SPLIT_MANIFEST.json'
    split = read(splitpath)
    assert inputs[str(splitpath)] == manifest['split_manifest_sha256']
    monitorpath = ROOT.parent/'monitoring/SCHEDULED_ROUND09_TERMINAL_20261004T0409.json'
    read(monitorpath)
    assert inputs[str(monitorpath)] == MONITOR
    cfg = read(ROOT/'config.json')
    assert cfg['promotion_delta_r2'] == .003
    assert cfg['retained_v2_reference_r2'] == .44716940136585204
    roster = read(ROOT/'PARAMETER_ROSTER.json')
    assert inputs[str(ROOT/'PARAMETER_ROSTER.json')] == manifest['parameter_roster_sha256']
    assert len(roster['contract']['included']) == 135
    assert len(roster['contract']['excluded_frozen']) == 11
    opaque = {}
    arms = {}
    counts = set()
    for arm, pid, gpu, decay in [('zero_decay', 2004444, 1, 0.), ('coupled_l2', 2004449, 2, 1e-4)]:
        rd = ROOT/'runs'/arm
        assert not (rd/'FAILED.json').exists() and not list(rd.glob('RESUMED_FAILURE_*'))
        terminal = read(rd/'FIT_COMPLETE.json')
        assert read(rd/'status.json') == dict(terminal, state='complete')
        assert terminal['arm'] == arm and terminal['completed_epoch'] == 60
        assert terminal['steps'] == 112860 and terminal['test_evaluated'] is False
        assert terminal['manifest_sha256'] == MANIFEST
        assert terminal['source_split_manifest_sha256'] == manifest['split_manifest_sha256']
        assert terminal['initial_base_tensor_sha256'] == manifest['initial_base_tensor_sha256']
        hp = rd/'history.jsonl'; inputs[str(hp)] = sha(hp)
        assert inputs[str(hp)] == terminal['history_sha256']
        history = [json.loads(line) for line in hp.read_text().splitlines()]
        assert [r['epoch'] for r in history] == list(range(61))
        assert history[0]['order_sha256'] is None
        for row in history:
            val = row['validation']; pooled = val['pooled']
            assert pooled['count'] == 66860
            near(pooled['mse'], pooled['sse']/66860)
            near(pooled['rmse']**2, pooled['mse'])
            assert set(val['per_state']) == {str(i) for i in range(1, 11)}
            assert all(v['count'] == 6686 for v in val['per_state'].values())
            near(sum(v['sse'] for v in val['per_state'].values()), pooled['sse'])
            assert sum(v['count'] for v in val['false_bright_bins'].values()) == 66860
            near(sum(v['sse'] for v in val['false_bright_bins'].values()), pooled['sse'])
            assert val['bright_tail']['q90']['threshold'] == .0549
            assert val['bright_tail']['q99']['threshold'] == .2406
            if not row['epoch']:
                continue
            assert row['order_sha256'] == manifest['epoch_order_sha256'][row['epoch']-1]
            assert row['optimizer_batches'] == 1881 and row['learning_rate'] == .001
            assert row['weight_decay'] == decay
            near(row['train']['total'], row['train']['base'])
            near(row['train']['base'], row['train']['energy']+row['train']['trace'])
            assert 0 <= row['gradient_clip_fraction'] <= 1
            d = row['transport_diagnostics']
            assert all(math.isfinite(v) for v in d.values())
            assert row['gate_parameter_movement_l2'] == d['gate_gradient_l2'] == 0
            for block in (2, 3):
                assert d[f'block{block}_coefficient_min'] == d[f'block{block}_coefficient_max'] == d[f'block{block}_theta_l2_mean'] == 0
                for l in (1, 2, 3):
                    p = f'block{block}_l{l}_'
                    for field in ('delta_norm', 'original_message_norm', 'preblock_T_norm', 'regularized_relative_to_message', 'regularized_relative_to_preblock'):
                        n = d[p+field+'_count']; counts.add(n)
                        assert isinstance(n, int) and n > 120355
                        near(d[p+field+'_mean'], d[p+field+'_sum']/n)
                    assert d[p+'delta_norm_sum'] == d[p+'delta_norm_max'] == 0
                    assert d[p+'regularized_relative_to_message_max'] == d[p+'regularized_relative_to_preblock_max'] == 0
            reg = row['regularization_diagnostics']
            assert all(math.isfinite(v) for v in reg.values())
            assert reg['parameters_with_grad'] == 135 and reg['parameters_grad_none'] == 0
            assert 0 <= reg['task_norm_zero'] <= reg['task_norm_below_floor'] <= 1881
            near(reg['coupled_term_l2'], decay*reg['trainable_parameter_l2'])
            assert reg['radial_beta_min_abs'] > 0 and reg['energy_offset_min'] <= reg['energy_offset_max']
            if not decay:
                assert reg['coupled_term_l2'] == reg['regularized_decay_task_ratio'] == 0
                near(reg['effective_gradient_l2'], reg['postclip_task_gradient_l2'])
        chosen = min(history, key=lambda r: r['validation']['pooled']['sse'])
        best = terminal['best']
        assert read(rd/'BEST.json') == best
        assert chosen['epoch'] == best['epoch']
        assert chosen['validation']['pooled']['sse'] == best['sse']
        assert chosen['validation']['pooled']['r2'] == best['r2']
        for p, h in [(rd/'last.pt', terminal['last_checkpoint_sha256']),
                     (rd/'best.pt', terminal['best_checkpoint_sha256']),
                     (rd/'geometry_best.pt', terminal['geometry_checkpoint_sha256']),
                     (rd/best['checkpoint'], best['checkpoint_sha256']),
                     (rd/best['predictions'], best['predictions_sha256'])]:
            assert p.resolve().is_relative_to(rd) and sha(p) == h
            opaque[str(p)] = h
        assert best['checkpoint_sha256'] == terminal['best_checkpoint_sha256']
        launches = list((rd/'attempts').glob('*/LAUNCH_RECEIPT.json'))
        assert len(launches) == 1
        launch = read(launches[0])
        assert launch['arm'] == arm and launch['gpu'] == gpu and launch['identity']['pid'] == pid
        assert launch['registration_returncode'] == 0 and launch['registration_barrier_released'] is True
        assert launch['manifest_sha256'] == MANIFEST and launch['authorization_sha256'] == AUTH
        assert launch['review_sha256'] == auth['independent_review_sha256']
        assert launch['publication_sha256'] == auth['publication_receipt_sha256']
        assert launch['environment']['CUDA_VISIBLE_DEVICES'] == launch['gpu_uuid']
        registration = read(launches[0].parent/'REGISTRATION_TOOL.log')
        for key in ('pid', 'start_ticks', 'uid', 'cwd', 'argv_sha256', 'boot_id'):
            assert registration['identity'][key] == launch['identity'][key]
        with (launches[0].parent/'train.log').open() as f:
            barrier = json.loads(f.readline())['registration_barrier_passed']
        assert barrier == {'registered': True, 'pid': pid, 'binding_sha256': AUTH}
        access = read(rd/'DATA_ACCESS.json')
        assert access['test_numeric_rows'] == 0
        for name, key in [('train', 'train'), ('validation', 'val')]:
            assert len(access[name]) == 12
            for item in access[name]:
                assert item['partition'] == key and item['numeric_rows'] == split['counts'][key]
                assert item['numeric_global_indices_sha256'] == split['arrays'][key+'_indices.npy']['content_sha256']
        arms[arm] = {'epochs': 60, 'steps': 112860, 'history_rows': 61,
                     'selected_epoch': best['epoch'], 'selected_r2': best['r2'],
                     'fixed60_r2': history[60]['validation']['pooled']['r2'],
                     'weight_decay': decay, 'all_orders_match': True, 'original_mode_and_dormant_branch': True,
                     'one_original_attempt_no_failure_or_resume': True, 'all66860_labels_every_epoch': True,
                     'all_states_and_bins_partition_pooled': True, 'test_numeric_rows': 0}
    assert len(counts) == 1
    delta = arms['coupled_l2']['selected_r2']-arms['zero_decay']['selected_r2']
    retained_delta = arms['coupled_l2']['selected_r2']-.44716940136585204
    inputs[str(Path(__file__))] = sha(__file__)
    for p, h in inputs.items():
        assert sha(p) == h
    result = {'passed': True, 'scope': 'terminal_text_aggregate_sources_and_opaque_hashes_only',
              'source_files_verified': 360, 'arms': arms, 'consistent_train_atom_count': next(iter(counts)),
              'candidate_minus_current_control': delta, 'candidate_minus_retained_reference': retained_delta,
              'dual_003_allocation_gate': delta >= .003 and retained_delta >= .003,
              'input_hashes': inputs, 'opaque_checkpoint_prediction_hashes_only': opaque,
              'no_checkpoint_tensor_or_prediction_array_decode_by_reviewer': True,
              'no_model_inference_or_TEST_access': True, 'no_duplicate_live_process_probe': True,
              'owned_process_absence_evidence_owner': 'QC exact terminal receipt',
              'checkpoint_tensor_and_saved_array_reduction_owner': 'science, one reviewed CPU analysis'}
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'dual_gate': result['dual_003_allocation_gate']}))


if __name__ == '__main__':
    main()
