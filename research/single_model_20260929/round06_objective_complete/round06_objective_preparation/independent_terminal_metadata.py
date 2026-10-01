"""Independent terminal JSON/source and opaque-hash checks; no model/data decode."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MANIFEST = '4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b'
EXPECTED_AUTH = '2c6c7f75993719fb5386df162248461cd7355bd2f033ae0bfd2848c146d9d41c'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), (a, b)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT/'completion/INDEPENDENT_TERMINAL_METADATA.json'
    assert not output.exists(), 'Preserve completed independent review'
    manifest_path = ROOT/'FROZEN_MANIFEST.json'
    assert sha(manifest_path) == EXPECTED_MANIFEST
    manifest = read(manifest_path)
    for path, digest in manifest['source_hashes'].items():
        assert sha(path) == digest, path
    authpath = ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json'
    assert sha(authpath) == EXPECTED_AUTH
    auth = read(authpath)
    assert auth['authorized'] and auth['scope'] == 'round06_two_arm_60epoch_fit'
    assert auth['arms'] == ['trace_control', 'raw_f'] and auth['epochs'] == 60
    assert not auth['test_access'] and not auth['historical_weights']
    assert auth['frozen_manifest_sha256'] == EXPECTED_MANIFEST
    assert sha(ROOT/'INDEPENDENT_PREPARATION_REVIEW.json') == auth['independent_review_sha256']
    assert sha(auth['publication_receipt']) == auth['publication_receipt_sha256']
    pub = read(auth['publication_receipt'])
    assert pub['remote_verified'] and pub['download_before_stage_before_commit_push']
    assert pub['frozen_manifest_sha256'] == EXPECTED_MANIFEST
    splitpath = ROOT.parent/'dataset_audit_20260930/SPLIT_MANIFEST.json'
    split = read(splitpath)
    assert sha(splitpath) == manifest['split_manifest_sha256']
    inputs = {str(p): sha(p) for p in [manifest_path, authpath, Path(auth['publication_receipt']), splitpath, Path(__file__)]}
    opaque = {}
    arms = {}
    for arm, expected_pid, expected_gpu in [('trace_control', 1450628, 1), ('raw_f', 1450633, 2)]:
        rd = ROOT/'runs'/arm
        assert not (rd/'FAILED.json').exists()
        terminalpath = rd/'FIT_COMPLETE.json'
        terminal = read(terminalpath)
        status = read(rd/'status.json')
        assert terminal['arm'] == arm and terminal['completed_epoch'] == 60
        assert terminal['steps'] == 112860 and not terminal['test_evaluated']
        assert terminal['manifest_sha256'] == EXPECTED_MANIFEST
        assert terminal['source_split_manifest_sha256'] == manifest['split_manifest_sha256']
        assert terminal['initial_base_tensor_sha256'] == manifest['initial_base_tensor_sha256']
        assert status == dict(terminal, state='complete')
        historypath = rd/'history.jsonl'
        assert sha(historypath) == terminal['history_sha256']
        history = [json.loads(line) for line in historypath.read_text().splitlines()]
        assert [row['epoch'] for row in history] == list(range(61))
        for row in history:
            val = row['validation']; pooled = val['pooled']
            assert pooled['count'] == 66860
            near(pooled['mse'], pooled['sse']/66860)
            near(pooled['rmse']**2, pooled['mse'])
            states = val['per_state']
            assert set(states) == {str(i) for i in range(1, 11)}
            assert all(x['count'] == 6686 for x in states.values())
            near(sum(x['sse'] for x in states.values()), pooled['sse'])
            bins = val['false_bright_bins']
            assert sum(x['count'] for x in bins.values()) == 66860
            near(sum(x['sse'] for x in bins.values()), pooled['sse'])
            assert val['bright_tail']['q90']['threshold'] == .0549
            assert val['bright_tail']['q99']['threshold'] == .2406
            if row['epoch']:
                assert row['order_sha256'] == manifest['epoch_order_sha256'][row['epoch']-1]
                assert row['optimizer_batches'] == 1881 and row['learning_rate'] == .001
                assert row['adapter_parameter_movement_l2'] == row['F_relative_update_max'] == 0
                train = row['train']
                assert train['F_relative_update_mean'] == 0
                near(train['base'], train['energy']+train['trace'])
                intensity = train['trace'] if arm == 'trace_control' else train['normalized_raw_f']
                near(train['total'], train['energy']+intensity)
        chosen = min(history, key=lambda row: row['validation']['pooled']['sse'])
        best = terminal['best']
        assert chosen['epoch'] == best['epoch']
        assert chosen['validation']['pooled']['sse'] == best['sse']
        assert chosen['validation']['pooled']['r2'] == best['r2']
        files = [(rd/'last.pt', terminal['last_checkpoint_sha256']),
                 (rd/'best.pt', terminal['best_checkpoint_sha256']),
                 (rd/'geometry_best.pt', terminal['geometry_checkpoint_sha256']),
                 (rd/best['checkpoint'], best['checkpoint_sha256']),
                 (rd/best['predictions'], best['predictions_sha256'])]
        for path, digest in files:
            assert path.resolve().is_relative_to(rd) and sha(path) == digest
            opaque[str(path)] = digest
        assert terminal['best_checkpoint_sha256'] == best['checkpoint_sha256']
        launches = list((rd/'attempts').glob('*/LAUNCH_RECEIPT.json'))
        assert len(launches) == 1
        launch = read(launches[0])
        assert launch['arm'] == arm and launch['gpu'] == expected_gpu
        assert launch['identity']['pid'] == expected_pid
        assert not Path('/proc', str(expected_pid)).exists(), 'Original PID must be absent at this check'
        assert launch['registration_returncode'] == 0
        assert launch['manifest_sha256'] == EXPECTED_MANIFEST
        assert launch['authorization_sha256'] == EXPECTED_AUTH
        assert launch['review_sha256'] == auth['independent_review_sha256']
        assert launch['publication_sha256'] == auth['publication_receipt_sha256']
        assert launch['environment']['CUDA_VISIBLE_DEVICES'] == launch['gpu_uuid']
        access = read(rd/'DATA_ACCESS.json')
        assert access['test_numeric_rows'] == 0
        for name, key in [('train', 'train'), ('validation', 'val')]:
            assert len(access[name]) == 12
            for item in access[name]:
                assert item['partition'] == key and item['numeric_rows'] == split['counts'][key]
                assert item['numeric_global_indices_sha256'] == split['arrays'][key+'_indices.npy']['content_sha256']
        for path in [terminalpath, rd/'status.json', historypath, rd/'DATA_ACCESS.json', launches[0]]:
            inputs[str(path)] = sha(path)
        arms[arm] = {'epochs': 60, 'steps': 112860, 'history_rows': 61,
                     'selected_epoch': best['epoch'], 'selected_r2': best['r2'],
                     'fixed60_r2': history[60]['validation']['pooled']['r2'],
                     'selected_pooled_count': 66860, 'all_orders_match': True,
                     'one_original_attempt_and_pid_absent': True, 'no_failure_marker': True,
                     'test_numeric_rows': 0, 'all_state_and_brightness_bins_partition_pooled': True,
                     'dormant_F_unchanged_all_epochs': True}
    delta = arms['raw_f']['selected_r2']-arms['trace_control']['selected_r2']
    retained_delta = arms['raw_f']['selected_r2']-.44716940136585204
    result = {'passed': True, 'scope': 'terminal_json_sources_selection_and_opaque_hashes_only',
              'source_files_verified': len(manifest['source_hashes']), 'arms': arms,
              'candidate_minus_current_control': delta, 'candidate_minus_retained_reference': retained_delta,
              'dual_003_allocation_gate': delta >= .003 and retained_delta >= .003,
              'input_hashes': inputs, 'opaque_checkpoint_prediction_hashes_only': opaque,
              'no_checkpoint_tensor_or_prediction_array_decode_by_reviewer': True,
              'no_model_inference_or_TEST_access': True,
              'deeper_checkpoint_state_and_saved_metric_recomputation_owner': 'science; independent source/output review follows'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(output), 'dual_gate': result['dual_003_allocation_gate']}))


if __name__ == '__main__':
    main()
