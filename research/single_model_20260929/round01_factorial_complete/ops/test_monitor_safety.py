#!/usr/bin/env python3
"""Safety checks for identity and containment; no training job is signalled."""
import json
import os
import tempfile
from pathlib import Path
from monitor import ROOT, gpu_snapshot, inspect_run, process_identity, safe_path

checks = {}
identity = process_identity(os.getpid())
checks['own_identity_readable'] = identity['pid'] == os.getpid() and identity['start_ticks'] > 0
for escape in ('../outside.json', '/etc/passwd'):
    try:
        safe_path(ROOT, escape)
        checks['reject_' + escape] = False
    except ValueError:
        checks['reject_' + escape] = True
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    run = {'id': 'fixture', 'run_dir': '.', 'identity': identity}
    checks['running_identity'] = inspect_run(root, run)['state'] == 'RUNNING_IDENTITY_MATCH'
    run['identity'] = dict(identity, start_ticks=identity['start_ticks'] + 1)
    checks['reject_reused_pid'] = inspect_run(root, run)['state'] == 'PID_REUSED_OR_IDENTITY_MISMATCH'
    run['identity'] = dict(identity, pid=2147483647)
    checks['absent_process'] = inspect_run(root, run)['state'] == 'EXITED_WITHOUT_COMPLETION_REVIEW_REQUIRED'
    (root / 'complete.json').write_text('{}')
    run['completion_path'] = 'complete.json'
    checks['completion_recorded'] = inspect_run(root, run)['state'] == 'COMPLETE_MARKER_PRESENT'
    (root / 'failed.json').write_text('{}')
    run['failure_path'] = 'failed.json'
    checks['failure_not_silenced'] = inspect_run(root, run)['state'] == 'FAILURE_MARKER_REVIEW_REQUIRED'
gpus = gpu_snapshot()['gpus']
checks['eight_gpus_detected'] = len(gpus) == 8
checks['ecc_fields_detected'] = all('dram_uncorrectable' in g['ecc_volatile'] for g in gpus)
checks['known_unhealthy_gpu_detected'] = int(gpus[3]['ecc_volatile']['dram_uncorrectable']) > 0
print(json.dumps({'checks': checks, 'passed': all(checks.values()), 'test_kind': 'read-only identity/containment/health safety checks'}, indent=2))
assert all(checks.values())
