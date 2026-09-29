#!/usr/bin/env python3
"""One registered fixed33 source startup check; no validation or weight loading."""
import json
import subprocess
from monitor import ROOT, atomic_json, file_info, process_identity, stamp

RUN_ID = 'round03_source33_1790707949921845597'
PID = 1174422
GPU_UUID = 'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'
MANIFEST = '93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37'

def main():
    call = subprocess.run(['/usr/bin/python3', str(ROOT / 'ops/monitor.py'), '--root', str(ROOT)],
                          text=True, capture_output=True, check=True)
    snapshot = json.loads(call.stdout)['record']
    report = json.loads((ROOT / 'monitoring' / snapshot).read_text())
    registry = json.loads((ROOT / 'ops/registry.json').read_text())
    registered = next(r for r in registry['runs'] if r['id'] == RUN_ID)
    observed = next(r for r in report['runs'] if r['id'] == RUN_ID)
    assert registered['identity']['pid'] == PID and registered['gpu_uuid'] == GPU_UUID
    assert registered['run_dir'] == 'round03_transfer/runs/source33'
    rd = ROOT / registered['run_dir']
    identity = process_identity(PID)
    complete = (rd / 'FIT_COMPLETE.json').exists()
    keys = ('pid', 'start_ticks', 'uid', 'cwd', 'argv_sha256', 'boot_id')
    matches = identity is not None and all(identity[k] == registered['identity'][k] for k in keys)
    assert matches or (identity is None and complete), 'Owned identity/terminal state requires review'
    assert not (rd / 'FAILED.json').exists()
    if matches and not complete:
        assert observed['state'] == 'RUNNING_IDENTITY_MATCH'
        assert observed['observed_gpu_uuids'] == [GPU_UUID]
    assert report['gpu_snapshot']['query_returncode'] == 0
    gpus = {g['index']: g for g in report['gpu_snapshot']['gpus']}
    gpu = gpus[1]
    assert gpu['uuid'] == GPU_UUID
    ecc = [v for group in ('ecc_volatile', 'ecc_aggregate') for k, v in gpu[group].items() if 'uncorrectable' in k]
    assert ecc and all(str(v).isdigit() and int(v) == 0 for v in ecc)
    assert gpu['remapped_rows']['remapped_row_pending'] == 'No'
    assert gpu['remapped_rows']['remapped_row_failure'] == 'No'
    rows = [json.loads(line) for line in (rd / 'history.jsonl').read_text().splitlines() if line.strip()]
    assert rows and [row['epoch'] for row in rows] == list(range(1, rows[-1]['epoch'] + 1))
    assert all('validation' not in row and 'train' in row for row in rows)
    access = json.loads((rd / 'TRAIN_DATA_ACCESS.json').read_text())
    assert access['global_rows'] == 96284 and access['calibration_validation_test_target_rows_decoded'] == 0
    assert access['raw_f_decoded'] is False
    assert access['fit_index_sha256'] == 'b7b00dfe514ae57f3dd609a60dc259625007a5f43288ff75e2c83a3a7e71da37'
    status = json.loads((rd / 'status.json').read_text())
    assert status['manifest_sha256'] == MANIFEST
    last = file_info(rd / 'last.pt')
    assert last['exists'] and last['bytes'] > 0
    unrelated_present = any(p['pid'] == '712998' for p in gpus[0]['compute_processes'])
    assert unrelated_present, 'Record unrelated owner change without intervention'
    result = {'passed': True, 'checked_at_utc': stamp(), 'monitor_record': snapshot,
        'run_id': RUN_ID, 'pid': PID, 'physical_gpu': 1, 'gpu_uuid': GPU_UUID,
        'pinned_identity': registered['identity'], 'observed_identity': identity, 'identity_matches': matches,
        'authoritative_state': observed['state'], 'manifest_sha256': MANIFEST,
        'first_epoch_seconds': rows[0]['seconds'], 'latest_complete_epoch': rows[-1]['epoch'],
        'latest_epoch_seconds': rows[-1]['seconds'], 'latest_training_losses': rows[-1]['train'],
        'fixed_source_epochs': 33, 'source_progress_has_no_validation_fields': True,
        'data_access_metadata': access, 'last_checkpoint_stat_only': last,
        'source_final_stat_only': file_info(rd / 'source_final.pt'),
        'source_final_not_expected_until_epoch33': not complete,
        'unrelated_gpu0_preserved': unrelated_present, 'checkpoint_contents_loaded': False,
        'four_hour_schedule_unchanged': True, 'interventions': []}
    atomic_json(ROOT / 'ops/ROUND03_STARTUP_CHECK.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
