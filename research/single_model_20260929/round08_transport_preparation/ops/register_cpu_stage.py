#!/usr/bin/env python3
"""Register a live, owned CPU stage wrapper without changing sealed GPU tooling."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, process_identity, safe_path, stamp

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=ROOT)
p.add_argument('--id', required=True)
p.add_argument('--pid', type=int, required=True)
p.add_argument('--run-dir', required=True)
p.add_argument('--status-path', required=True)
p.add_argument('--log-path', required=True)
p.add_argument('--completion-path', required=True)
p.add_argument('--failure-path', required=True)
p.add_argument('--authorization', required=True)
p.add_argument('--authorization-sha256', required=True)
p.add_argument('--checkpoint-path', action='append', default=[])
a = p.parse_args()
root = a.root.resolve()
run_dir = safe_path(root, a.run_dir)
if not run_dir.is_dir():
    raise SystemExit('CPU stage directory must exist within the owned scope')
paths = {key: safe_path(root, getattr(a,key)) for key in
         ['status_path','log_path','completion_path','failure_path','authorization']}
for path in a.checkpoint_path:
    safe_path(root,path)
if paths['completion_path'].exists() or paths['failure_path'].exists():
    raise SystemExit('Use a fresh attempt with no terminal marker')
auth_sha = hashlib.sha256(paths['authorization'].read_bytes()).hexdigest()
if auth_sha != a.authorization_sha256:
    raise SystemExit('Execution authorization hash mismatch')
identity = process_identity(a.pid)
if not identity or identity['uid'] != root.stat().st_uid or identity['process_state']=='Z':
    raise SystemExit('CPU wrapper is absent, exited, or belongs to another account')
if not Path(identity['cwd']).is_relative_to(root):
    raise SystemExit('CPU wrapper cwd must be inside this research scope')
# Read only the required environment entries; never emit the full environment.
selected={}
for item in (Path('/proc')/str(a.pid)/'environ').read_bytes().split(b'\0'):
    key, sep, value = item.partition(b'=')
    if sep and key in [b'CUDA_VISIBLE_DEVICES',b'OMP_NUM_THREADS',b'MKL_NUM_THREADS']:
        selected[key.decode()] = value.decode()
if selected != {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}:
    raise SystemExit('CPU wrapper must start with empty CUDA visibility and two CPU threads')
keys=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
again=process_identity(a.pid)
if not again or any(again[k]!=identity[k] for k in keys):
    raise SystemExit('CPU wrapper identity changed during registration')
run={'id':a.id,'registered_at_utc':stamp(),'identity':identity,'run_dir':a.run_dir,
     'resource_kind':'cpu','gpu_uuid':None,'expected_gpu_allocations':0,
     'environment_constraints':selected,'authorization_path':a.authorization,
     'authorization_sha256':auth_sha,'status_path':a.status_path,'log_path':a.log_path,
     'completion_path':a.completion_path,'failure_path':a.failure_path,
     'checkpoint_paths':a.checkpoint_path,'stale_after_seconds':28800,
     'monitor':True,'restart_policy':'coordinator_review_required',
     'identity_role':'synchronous owned stage wrapper; child identities recorded in stage receipt'}
with (root/'ops/registry.lock').open('w') as lock:
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    registry=json.loads((root/'ops/registry.json').read_text())
    if any(r['id']==a.id for r in registry['runs']):
        raise SystemExit('CPU attempt ID already exists; never replace a registered attempt')
    registry['runs'].append(run)
    atomic_json(root/'ops/registry.json',registry)
    atomic_json(run_dir/'MONITOR_REGISTRATION.json',run)
print(json.dumps(run,indent=2))
