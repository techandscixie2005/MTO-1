#!/usr/bin/env python3
"""Read-only, explicitly registered MTO run monitoring; safe for cron."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929')


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def command(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=40)
        return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except Exception as e:
        return {'returncode': -1, 'stdout': '', 'stderr': repr(e)}


def process_identity(pid):
    p = Path('/proc') / str(int(pid))
    try:
        stat = (p / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'uid': p.stat().st_uid, 'cwd': str((p / 'cwd').resolve(strict=True)),
                'argv_sha256': hashlib.sha256((p / 'cmdline').read_bytes()).hexdigest(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'process_state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def safe_path(root, value):
    p = (root / value).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Path escapes owned research scope: ' + value)
    return p


def file_info(path):
    if not path.is_file():
        return {'exists': False}
    s = path.stat()
    return {'exists': True, 'bytes': s.st_size, 'mtime_utc': dt.datetime.fromtimestamp(s.st_mtime, dt.timezone.utc).isoformat(),
            'age_seconds': round(time.time() - s.st_mtime, 1)}


def gpu_snapshot():
    xml = command(['/usr/bin/nvidia-smi', '-q', '-x'])
    result = {'query_returncode': xml['returncode'], 'query_error': xml['stderr'], 'gpus': []}
    if xml['returncode']:
        return result
    try:
        tree = ET.fromstring(xml['stdout'])
    except ET.ParseError as error:
        result['query_returncode'] = -2
        result['query_error'] = 'NVIDIA XML could not be parsed: ' + str(error)
        return result
    for index, g in enumerate(tree.findall('gpu')):
        def val(path):
            return g.findtext(path, default='unknown')
        result['gpus'].append({'index': index, 'uuid': val('uuid'), 'name': val('product_name'),
            'temperature_c': val('temperature/gpu_temp'), 'memory_used': val('fb_memory_usage/used'),
            'utilization': val('utilization/gpu_util'),
            'ecc_volatile': {n.tag: n.text for n in g.findall('ecc_errors/volatile/*') if len(n) == 0},
            'ecc_aggregate': {n.tag: n.text for n in g.findall('ecc_errors/aggregate/*') if len(n) == 0},
            'remapped_rows': {n.tag: n.text for n in g.findall('remapped_rows/*') if len(n) == 0},
            'compute_processes': [{'pid': p.findtext('pid'), 'name': p.findtext('process_name'),
                'used_memory': p.findtext('used_memory')} for p in g.findall('processes/process_info')]})
    return result


def inspect_run(root, run):
    answer = {'id': run['id'], 'run_dir': run['run_dir'], 'gpu_uuid': run.get('gpu_uuid'), 'restart_action': 'none'}
    observed = process_identity(run['identity']['pid'])
    answer['process_identity'] = observed
    keys = ('pid', 'start_ticks', 'uid', 'cwd', 'argv_sha256', 'boot_id')
    answer['identity_matches'] = observed is not None and all(observed.get(k) == run['identity'].get(k) for k in keys)
    files = {}
    for kind in ('status_path', 'log_path', 'completion_path', 'failure_path'):
        if run.get(kind):
            path = safe_path(root, run[kind])
            files[kind] = file_info(path)
            if files[kind]['exists'] and kind != 'log_path' and path.stat().st_size <= 128 * 1024:
                try:
                    files[kind]['content'] = json.loads(path.read_text())
                except (ValueError, UnicodeDecodeError):
                    files[kind]['content_parse_error'] = True
    answer['files'] = files
    answer['checkpoints'] = {p: file_info(safe_path(root, p)) for p in run.get('checkpoint_paths', [])}
    complete = files.get('completion_path', {}).get('exists', False)
    failure = files.get('failure_path', {}).get('exists', False)
    if observed and not answer['identity_matches']:
        state = 'PID_REUSED_OR_IDENTITY_MISMATCH'
    elif failure:
        state = 'FAILURE_MARKER_REVIEW_REQUIRED'
    elif complete:
        state = 'COMPLETE_MARKER_PRESENT'
    elif observed and observed['process_state'] != 'Z':
        ages = [f['age_seconds'] for k, f in files.items() if k in ('status_path', 'log_path') and f.get('exists')]
        registered = dt.datetime.fromisoformat(run.get('registered_at_utc', stamp()))
        age_without_files = (dt.datetime.now(dt.timezone.utc) - registered).total_seconds()
        stale = (min(ages) if ages else age_without_files) > run.get('stale_after_seconds', 28800)
        state = 'RUNNING_PROGRESS_STALE' if stale else 'RUNNING_IDENTITY_MATCH'
    else:
        state = 'EXITED_WITHOUT_COMPLETION_REVIEW_REQUIRED'
    answer['state'] = state
    if ('REQUIRED' in state or 'MISMATCH' in state or 'STALE' in state) and run.get('log_path'):
        log = safe_path(root, run['log_path'])
        if log.is_file():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 8192))
                answer['diagnostic_log_tail'] = f.read().decode('utf-8', errors='replace')
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'monitoring'
    out.mkdir(exist_ok=True)
    with (out / 'monitor.lock').open('w') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        registry = json.loads((root / 'ops/registry.json').read_text())
        previous = json.loads((out / 'latest.json').read_text()) if (out / 'latest.json').exists() else {}
        report = {'schema_version': 1, 'checked_at_utc': stamp(), 'monitor_pid': os.getpid(),
                  'cadence_hours': 4, 'mode': 'read_only_no_signals_no_restarts',
                  'registry_sha256': hashlib.sha256((root / 'ops/registry.json').read_bytes()).hexdigest(),
                  'gpu_snapshot': gpu_snapshot(), 'runs': [], 'incidents': [], 'interventions': []}
        prev_states = {r['id']: r['state'] for r in previous.get('runs', [])}
        for run in registry['runs']:
            if not run.get('monitor', True):
                continue
            try:
                item = inspect_run(root, run)
                observed_gpu_uuids = [g['uuid'] for g in report['gpu_snapshot']['gpus']
                                     if str(run['identity']['pid']) in [p['pid'] for p in g['compute_processes']]]
                item['observed_gpu_uuids'] = observed_gpu_uuids
                if observed_gpu_uuids and observed_gpu_uuids != [run['gpu_uuid']]:
                    item['state'] = 'GPU_ASSIGNMENT_MISMATCH_REVIEW_REQUIRED'
            except Exception as e:
                item = {'id': run.get('id'), 'state': 'MONITOR_ERROR_REVIEW_REQUIRED', 'error': repr(e)}
            report['runs'].append(item)
            if item['state'] != prev_states.get(item['id']):
                report['incidents'].append({'run_id': item['id'], 'previous_state': prev_states.get(item['id']),
                    'state': item['state'], 'action': 'Record transition; coordinator reviews failure recovery. No job altered.'})
        for g in report['gpu_snapshot']['gpus']:
            if g['uuid'] in registry.get('preferred_gpu_uuids', []):
                ecc_items = list(g['ecc_volatile'].items()) + list(g['ecc_aggregate'].items())
                uncorrected = [v for k, v in ecc_items if 'uncorrectable' in k]
                has_ecc = not uncorrected or any(not str(v).isdigit() or int(v) > 0 for v in uncorrected)
                remap = g['remapped_rows']
                if has_ecc or remap.get('remapped_row_pending') != 'No' or remap.get('remapped_row_failure') != 'No':
                    report['incidents'].append({'gpu_uuid': g['uuid'], 'state': 'PREFERRED_GPU_HEALTH_REVIEW_REQUIRED', 'action': 'Do not launch new jobs; coordinator assesses safe resumable recovery of owned jobs only.'})
        if report['gpu_snapshot']['query_returncode']:
            report['incidents'].append({'state': 'GPU_QUERY_FAILED_REVIEW_REQUIRED'})
        report['next_check_by_utc'] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)).isoformat()
        filename = 'CHECK_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
        atomic_json(out / filename, report)
        atomic_json(out / 'latest.json', report)
        with (out / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'checked_at_utc': report['checked_at_utc'], 'record': filename,
                                'run_states': {r['id']: r['state'] for r in report['runs']},
                                'incidents': report['incidents']}) + '\n')
        print(json.dumps({'record': filename, 'registered_runs': len(report['runs']), 'incidents': len(report['incidents'])}))


if __name__ == '__main__':
    main()
