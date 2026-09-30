#!/usr/bin/env python3
"""Annotate one existing read-only monitor observation; never run training/stages."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from monitor import ROOT, atomic_json, safe_path, stamp

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-record', required=True)
    parser.add_argument('--automation-record', required=True)
    args = parser.parse_args()
    path = safe_path(ROOT, 'monitoring/' + args.monitor_record)
    observed = read(path)
    assert observed['mode'] == 'read_only_no_signals_no_restarts'
    assert not observed['interventions'] and not observed['incidents']
    summaries = []
    for run in observed['runs']:
        assert run['state'] == 'COMPLETE_MARKER_PRESENT', run['id']
        assert run['process_identity'] is None, run['id']
        assert not run['observed_gpu_uuids'], run['id']
        assert all(meta['exists'] for meta in run['checkpoints'].values()), run['id']
        terminal = run['files']['completion_path']['content']
        summaries.append({'id': run['id'], 'run_dir': run['run_dir'], 'state': run['state'],
            'original_pid_observed': False, 'completed_epoch': terminal['completed_epoch'],
            'registered_gpu_uuid': run['gpu_uuid'], 'owned_gpu_process_observed': False,
            'checkpoint_stat_metadata': run['checkpoints']})
    rd = ROOT / 'round03_transfer'
    publication_path = ROOT / 'ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json'
    assert sha(publication_path) == 'ffac1aa2c625ecd2b3e933494eb84f0e8b99efc063f94fdf31b9343144b85c1c'
    publication = read(publication_path)
    assert publication['remote_verified'] and publication['download_before_stage_before_commit_push']
    assert publication['new_fit_authorized'] is False
    assert sha(rd / 'FROZEN_MANIFEST.json') == publication['frozen_manifest_sha256']
    manifest = read(rd / 'FROZEN_MANIFEST.json')
    assert sha(rd / 'PROTOCOL.md') == manifest['source_hashes'][str(rd / 'PROTOCOL.md')]
    prelaunch = read(rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json')
    assert prelaunch['passed'] and prelaunch['frozen_manifest_sha256'] == publication['frozen_manifest_sha256']
    assert sha(rd / 'INDEPENDENT_REVIEW_MANIFEST.json') == publication['independent_review_manifest_sha256']
    for name, expected in read(rd / 'INDEPENDENT_REVIEW_MANIFEST.json')['files'].items():
        assert sha(Path(name)) == expected, name
    assert sha(ROOT / 'current_state/ROUND03_DECISION.md') == publication['root_decision_sha256']
    validation = read(rd / 'affine/VALIDATION_COMPLETE.json')
    assert validation['manifest_sha256'] == publication['frozen_manifest_sha256']
    assert sha(rd / 'affine/COEFFICIENTS_FROZEN.json') == validation['coefficient_receipt_sha256']
    assert sha(rd / 'affine/EXPORT_COMPLETE.json') == validation['export_receipt_sha256']
    assert not validation['test_evaluated'] and not validation['prediction_averaging']
    assert len(list((rd / 'affine').glob('VALIDATION_ATTEMPT_*.json'))) == 1
    assert all(item['raw_f']['count'] == 66860 for item in validation['metrics'].values())
    cron = subprocess.run(['crontab', '-l'], check=True, capture_output=True, text=True).stdout
    scheduled = [line for line in cron.splitlines() if str(ROOT / 'ops/monitor.py') in line and not line.lstrip().startswith('#')]
    assert len(scheduled) == 1 and scheduled[0].startswith('0 */4 * * * ')
    automation = read(safe_path(ROOT, args.automation_record))
    assert automation['status'] == 'ACTIVE' and automation['four_hour_schedule']
    assert automation['target_thread_matches'] and automation['matching_automation_count'] == 1
    handoffs = ('current_state/COORDINATOR.md', 'current_state/monitor.md',
        'current_state/science_implementation.md', 'ops/ROUND03_RUNTIME_HANDOFF.md')
    stale = []
    for name in handoffs:
        content = (ROOT / name).read_text()
        for line_number, line in enumerate(content.splitlines(), 1):
            if 'Completion publication is owned by monitor and pending in this handoff.' in line:
                stale.append({'path': name, 'line': line_number, 'text': line,
                    'disposition': 'Flagged to coordinator; authoritative completion receipt supersedes this operational wording. No sealed snapshot edited.'})
    result = {'checked_at_utc': stamp(), 'kind': 'scheduled four-hour completed-campaign check',
        'monitor_record': str(path.relative_to(ROOT)), 'monitor_record_sha256': sha(path),
        'registered_runs': summaries, 'registered_run_count': len(summaries),
        'active_owned_jobs': 0, 'owned_failure_incidents': 0,
        'round03_protocol_review_decision_publication_bindings_verified': True,
        'source_epoch33_complete': True, 'coefficient_export_validation_complete': True,
        'pending_authorized_continuation': False, 'pending_round03_archive': False,
        'publication_commit': publication['commit'], 'completion_receipt_sha256': sha(publication_path),
        'validation_reopened': False, 'fits_launched_or_restarted': False,
        'checkpoint_tensors_or_raw_arrays_loaded': False, 'interventions': [],
        'schedules_unchanged': True, 'cron_command': scheduled[0], 'cron_sha256': hashlib.sha256(cron.encode()).hexdigest(),
        'matching_cron_count': 1, 'automation_verification': automation,
        'handoff_sha256': {name: sha(ROOT / name) for name in handoffs},
        'stale_operational_wording': stale, 'experiment_state_changed': False,
        'parent_notified_only_of_stale_operational_wording': True}
    token = observed['checked_at_utc'].replace('-', '').replace(':', '').split('.')[0].replace('+0000', 'Z')
    target = ROOT / 'monitoring' / ('CLOSED_CAMPAIGN_' + token + '.json')
    assert not target.exists()
    atomic_json(target, result)
    print(json.dumps({'record': target.relative_to(ROOT).as_posix(), 'registered_runs': len(summaries),
        'active_owned_jobs': 0, 'owned_failure_incidents': 0, 'pending_continuation_or_archive': False,
        'schedules_unchanged_and_single': True, 'stale_operational_paths': [r['path'] for r in stale]}, indent=2))

if __name__ == '__main__':
    main()
