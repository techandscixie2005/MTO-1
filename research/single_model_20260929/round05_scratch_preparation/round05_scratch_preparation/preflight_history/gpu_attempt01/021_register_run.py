#!/usr/bin/env python3
"""Register only a launch owned by this research scope, after launch receipt."""
import argparse
import fcntl
import json
from pathlib import Path
from monitor import ROOT, atomic_json, process_identity, safe_path, stamp

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=ROOT)
p.add_argument('--id', required=True)
p.add_argument('--pid', required=True, type=int)
p.add_argument('--run-dir', required=True)
p.add_argument('--gpu-uuid', required=True)
p.add_argument('--status-path', required=True)
p.add_argument('--log-path', required=True)
p.add_argument('--completion-path', required=True)
p.add_argument('--failure-path', required=True)
p.add_argument('--checkpoint-path', action='append', default=[])
a = p.parse_args()
root = a.root.resolve()
run_dir = safe_path(root, a.run_dir)
if not run_dir.is_dir():
    raise SystemExit('Run directory does not exist in the owned research scope')
identity = process_identity(a.pid)
if not identity or identity['uid'] != root.stat().st_uid:
    raise SystemExit('Process absent or not owned by research account')
cmdline = (Path('/proc') / str(a.pid) / 'cmdline').read_bytes().decode(errors='replace')
if str(root) not in cmdline and not Path(identity['cwd']).is_relative_to(root):
    raise SystemExit('Process argv or cwd must explicitly reference research scope')
run = {'id': a.id, 'registered_at_utc': stamp(), 'identity': identity, 'run_dir': a.run_dir,
       'gpu_uuid': a.gpu_uuid, 'status_path': a.status_path, 'log_path': a.log_path,
       'completion_path': a.completion_path, 'failure_path': a.failure_path,
       'checkpoint_paths': a.checkpoint_path, 'stale_after_seconds': 28800,
       'monitor': True, 'restart_policy': 'coordinator_review_required'}
for name in ('status_path', 'log_path', 'completion_path', 'failure_path'):
    safe_path(root, run[name])
for name in run['checkpoint_paths']:
    safe_path(root, name)
with (root / 'ops/registry.lock').open('w') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    registry = json.loads((root / 'ops/registry.json').read_text())
    if a.gpu_uuid not in registry['preferred_gpu_uuids']:
        raise SystemExit('GPU is not admitted as preferred healthy hardware')
    if any(r['id'] == a.id for r in registry['runs']):
        raise SystemExit('Run id exists; use a distinct attempt id for resumed attempts')
    registry['runs'].append(run)
    atomic_json(root / 'ops/registry.json', registry)
    atomic_json(run_dir / 'MONITOR_REGISTRATION.json', run)
print(json.dumps(run, indent=2))
