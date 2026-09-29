#!/usr/bin/env python3
"""One heartbeat observation of registered Round03 ownership and completion."""
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, file_info, gpu_snapshot, inspect_run, process_identity, stamp

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    rd = ROOT / 'round03_transfer'
    registry = json.loads((ROOT / 'ops/registry.json').read_text())
    attempts = [r for r in registry['runs'] if r['run_dir'] == 'round03_transfer/runs/source33']
    assert attempts, 'No owned source is registered'
    run = max(attempts, key=lambda r: r['registered_at_utc'])
    out = ROOT / run['run_dir']
    manifest = json.loads((rd / 'FROZEN_MANIFEST.json').read_text())
    manifest_hash = sha(rd / 'FROZEN_MANIFEST.json')
    review = json.loads((rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json').read_text())
    publication = json.loads((ROOT / 'ops/ROUND03_PUBLICATION_RECEIPT.json').read_text())
    launch = json.loads((out / 'LAUNCH_RECEIPT.json').read_text())
    assert review['passed'] and review['frozen_manifest_sha256'] == manifest_hash
    assert review['source_hashes'] == manifest['source_hashes']
    assert publication['remote_verified'] and publication['download_before_stage_before_commit_push']
    assert publication['frozen_manifest_sha256'] == manifest_hash
    assert publication['independent_review_sha256'] == sha(rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json')
    assert sha(rd / 'PROTOCOL.md') == manifest['source_hashes'][str(rd / 'PROTOCOL.md')]
    assert launch['attempt'] == run['id'] and launch['pid'] == run['identity']['pid']
    assert launch['manifest_sha256'] == manifest_hash and launch['monitor_registration_returncode'] == 0
    assert launch['publication_receipt_sha256'] == sha(ROOT / 'ops/ROUND03_PUBLICATION_RECEIPT.json')
    observed = inspect_run(ROOT, run)
    gpu_state = gpu_snapshot()
    assert gpu_state['query_returncode'] == 0, gpu_state['query_error']
    gpu = next(g for g in gpu_state['gpus'] if g['uuid'] == run['gpu_uuid'])
    uncorrected = [v for group in ('ecc_volatile', 'ecc_aggregate') for k, v in gpu[group].items() if 'uncorrectable' in k]
    healthy = bool(uncorrected) and all(str(v).isdigit() and int(v) == 0 for v in uncorrected)
    healthy &= gpu['remapped_rows'].get('remapped_row_pending') == 'No'
    healthy &= gpu['remapped_rows'].get('remapped_row_failure') == 'No'
    rows = [json.loads(s) for s in (out / 'history.jsonl').read_text().splitlines() if s.strip()]
    assert not rows or [r['epoch'] for r in rows] == list(range(1, rows[-1]['epoch'] + 1))
    assert all('validation' not in r and 'train' in r for r in rows)
    # Re-read terminal/identity after observations to avoid natural-exit races.
    current_identity = process_identity(run['identity']['pid'])
    terminal = json.loads((out / 'FIT_COMPLETE.json').read_text()) if (out / 'FIT_COMPLETE.json').exists() else None
    failed = (out / 'FAILED.json').exists()
    keys = ('pid', 'start_ticks', 'uid', 'cwd', 'argv_sha256', 'boot_id')
    identity_matches = current_identity is not None and all(current_identity[k] == run['identity'][k] for k in keys)
    original_exited = current_identity is None or not identity_matches or current_identity['process_state'] == 'Z'
    hashes, gate = {}, False
    if terminal:
        assert terminal['completed_epoch'] == 33 and terminal['manifest_sha256'] == manifest_hash
        assert terminal['fixed_epoch_selection'] and not terminal['heldout_or_outer_labels_used_during_fit']
        assert not terminal['test_evaluated']
        assert len(rows) == 33
        for name, key in [('source_final.pt', 'checkpoint_sha256'), ('last.pt', 'last_checkpoint_sha256')]:
            digest = sha(out / name)
            hashes[name] = {'sha256': digest, 'receipt_sha256': terminal[key], 'matches': digest == terminal[key],
                            **file_info(out / name)}
            assert hashes[name]['matches'], name
        gate = original_exited and healthy and not failed
        state = 'COMPLETE_VERIFIED_PROCESS_EXITED' if gate else 'TERMINAL_PRESENT_GATE_PENDING'
    elif failed:
        state = 'FAILURE_MARKER_REVIEW_REQUIRED'
    elif identity_matches and not original_exited:
        observed_uuids = [g['uuid'] for g in gpu_state['gpus']
            if str(run['identity']['pid']) in [p['pid'] for p in g['compute_processes']]]
        assert observed_uuids == [run['gpu_uuid']], observed_uuids
        state = 'RUNNING_IDENTITY_MATCH' if healthy else 'OWNED_GPU_HEALTH_REVIEW_REQUIRED'
    else:
        state = 'ABSENT_WITHOUT_TERMINAL_REVIEW_REQUIRED'
    # All actions are observational; recovery and affine execution belong to science/root.
    result = {'checked_at_utc': stamp(), 'kind': 'dedicated four-hour heartbeat observation',
        'run_id': run['id'], 'run_dir': run['run_dir'], 'registry_sha256': sha(ROOT / 'ops/registry.json'),
        'pinned_identity': run['identity'], 'observed_identity': current_identity,
        'identity_matches': identity_matches, 'original_source_process_exited': original_exited,
        'state': state, 'latest_complete_epoch': rows[-1]['epoch'] if rows else 0,
        'latest_epoch_seconds': rows[-1]['seconds'] if rows else None,
        'latest_train_losses': rows[-1]['train'] if rows else None,
        'source_fit_has_no_validation_fields': True, 'gpu': gpu, 'healthy_owned_gpu': bool(healthy),
        'unrelated_gpu0_observed_processes': gpu_state['gpus'][0]['compute_processes'],
        'status_log_checkpoint_metadata': observed['files'], 'checkpoints_stat_only': observed['checkpoints'],
        'terminal_record': terminal, 'completed_checkpoint_hashes': hashes,
        'completion_gate_verified_for_science': gate,
        'manifest_sha256': manifest_hash, 'review_sha256': sha(rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json'),
        'publication_receipt_sha256': sha(ROOT / 'ops/ROUND03_PUBLICATION_RECEIPT.json'),
        'context_hashes': {name: sha(ROOT / name) for name in ('current_state/COORDINATOR.md', 'ops/ROUND03_RUNTIME_HANDOFF.md')},
        'checkpoint_tensors_loaded': False, 'interventions': [], 'affine_commands_executed': False,
        'periodic_schedules_unchanged': True}
    token = result['checked_at_utc'].replace('-', '').replace(':', '').split('.')[0].replace('+0000', 'Z')
    destination = ROOT / 'monitoring' / ('ROUND03_HEARTBEAT_' + token + '.json')
    atomic_json(destination, result)
    print(json.dumps({'record': destination.relative_to(ROOT).as_posix(), 'state': state,
        'pid': run['identity']['pid'], 'start_ticks': run['identity']['start_ticks'],
        'identity_matches': identity_matches, 'latest_complete_epoch': result['latest_complete_epoch'],
        'latest_epoch_seconds': result['latest_epoch_seconds'], 'healthy_owned_gpu': bool(healthy),
        'original_source_process_exited': original_exited, 'completed_checkpoint_hashes': hashes,
        'completion_gate_verified_for_science': gate,
        'unrelated_gpu0_processes': result['unrelated_gpu0_observed_processes']}, indent=2))

if __name__ == '__main__':
    main()
