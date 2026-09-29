#!/usr/bin/env python3
"""One bounded authoritative Round02 startup check; no weight reads or signals."""
import hashlib
import json
from pathlib import Path
import subprocess
from monitor import ROOT, atomic_json, file_info, process_identity, stamp

EXPECTED = {
    'trace': {'id': 'round02_trace_1790705186282980177', 'pid': 1158193, 'gpu': 1,
              'uuid': 'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'},
    'raw_f': {'id': 'round02_raw_f_1790705186486661348', 'pid': 1158198, 'gpu': 2,
              'uuid': 'GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa'},
}

def main():
    check = subprocess.run(['/usr/bin/python3', str(ROOT / 'ops/monitor.py'), '--root', str(ROOT)],
                           text=True, capture_output=True, check=True)
    summary = json.loads(check.stdout)
    report = json.loads((ROOT / 'monitoring' / summary['record']).read_text())
    assert report['gpu_snapshot']['query_returncode'] == 0
    gpus = {g['index']: g for g in report['gpu_snapshot']['gpus']}
    registry = json.loads((ROOT / 'ops/registry.json').read_text())
    registered = {r['id']: r for r in registry['runs']}
    observed = {r['id']: r for r in report['runs']}
    result = []
    for arm, target in EXPECTED.items():
        run = registered[target['id']]
        item = observed[target['id']]
        assert run['identity']['pid'] == target['pid'] and run['gpu_uuid'] == target['uuid']
        actual = process_identity(target['pid'])
        rd = ROOT / run['run_dir']
        complete = (rd / 'FIT_COMPLETE.json').exists()
        assert not (rd / 'FAILED.json').exists(), arm
        keys = ('pid', 'start_ticks', 'uid', 'cwd', 'argv_sha256', 'boot_id')
        matches = actual is not None and all(actual.get(k) == run['identity'].get(k) for k in keys)
        assert matches or (actual is None and complete), (arm, 'Identity or terminal state requires review')
        if matches and not complete:
            assert item['state'] == 'RUNNING_IDENTITY_MATCH'
            assert item['observed_gpu_uuids'] == [target['uuid']]
        gpu = gpus[target['gpu']]
        assert gpu['uuid'] == target['uuid']
        ecc = [v for group in ('ecc_volatile', 'ecc_aggregate') for k, v in gpu[group].items()
               if 'uncorrectable' in k]
        assert ecc and all(str(v).isdigit() and int(v) == 0 for v in ecc)
        assert gpu['remapped_rows']['remapped_row_pending'] == 'No'
        assert gpu['remapped_rows']['remapped_row_failure'] == 'No'
        history = [json.loads(line) for line in (rd / 'history.jsonl').read_text().splitlines() if line.strip()]
        assert history[0]['epoch'] == 0
        last = history[-1]
        checkpoints = {name: file_info(rd / name) for name in ('best.pt', 'last.pt')}
        assert all(c['exists'] and c['bytes'] > 0 for c in checkpoints.values())
        result.append({'arm': arm, **target, 'pinned_identity': run['identity'],
                       'observed_identity': actual, 'identity_matches': matches,
                       'authoritative_monitor_state': item['state'],
                       'latest_complete_epoch': last['epoch'], 'seconds': last.get('seconds'),
                       'latest_raw_f_r2': last['validation']['raw_f']['r2'],
                       'best_epoch': last.get('best_epoch', 0),
                       'best_raw_f_r2': last.get('best_r2', history[0]['validation']['raw_f']['r2']),
                       'epoch_zero_raw_f_r2': history[0]['validation']['raw_f']['r2'],
                       'checkpoints_stat_only': checkpoints, 'healthy_gpu': True})
    assert len({r['pid'] for r in result}) == 2
    assert len({r['uuid'] for r in result}) == 2
    unaffected = any(p['pid'] == '712998' for p in gpus[0]['compute_processes'])
    assert unaffected, 'Unrelated GPU0 owner differs; record for coordination only'
    receipt = {'passed': True, 'checked_at_utc': stamp(), 'monitor_record': summary['record'],
               'type': 'one targeted startup check; periodic cadence unchanged',
               'runs': result, 'unrelated_gpu0_preserved': unaffected,
               'checkpoint_contents_loaded': False, 'interventions': [],
               'four_hour_schedule_unchanged': True}
    atomic_json(ROOT / 'ops/ROUND02_STARTUP_CHECK.json', receipt)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
