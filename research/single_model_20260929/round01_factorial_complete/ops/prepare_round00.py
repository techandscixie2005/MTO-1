#!/usr/bin/env python3
"""Create the reviewed initial-round source/record allowlist after source freeze."""
import hashlib
import json
from monitor import ROOT, atomic_json, stamp
from package_records import inspect

expected_manifest = 'eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3'
assert hashlib.sha256((ROOT/'FROZEN_MANIFEST.json').read_bytes()).hexdigest() == expected_manifest
frozen = json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
review = json.loads((ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json').read_text())
assert review['passed'] and review['frozen_manifest_sha256'] == expected_manifest
dependencies = json.loads((ROOT/'ops/DEPENDENCY_SOURCE_RECEIPT.json').read_text())['files']
for item in dependencies:
    assert frozen['source_hashes'][item['server_source']] == item['sha256']
    assert hashlib.sha256((ROOT/item['archived_source']).read_bytes()).hexdigest() == item['sha256']
files = '''train.py
metrics.py
summarize.py
round_config.json
freeze.py
launch.py
environment_capture.py
ENVIRONMENT.json
FROZEN_MANIFEST.json
GPU_SMOKE_ADMISSION.xml
GPU_SMOKE.json
gpu_smoke.log
gpu_smoke.py
history_baseline.md
response_operator_candidate.md
PROTOCOL.md
SECONDARY_CALIBRATION_POLICY.md
INDEPENDENT_PRELAUNCH_REVIEW.json
PRE_PILOT_DECISION_AMENDMENT.md
architecture/model.py
architecture/preflight.py
architecture/PREFLIGHT.json
architecture/preflight.log
baselines/calibrated_eta0.py
baselines/export_and_verify.py
baselines/README.md
baselines/CALIBRATED_EXPORT_VERIFICATION.json
baselines/export.log
baselines/export_initial_own_context_guard.log
baselines/export_initial_graphics_guard.log
monitoring/CHECK_20260929T155302Z.json
monitoring/CHECK_20260929T155359Z.json
monitoring/CHECK_20260929T160002Z.json
monitoring/cron.log
monitoring/events.jsonl
monitoring/latest.json
round00_audit/COORDINATOR.md
round00_audit/monitor.md'''.splitlines()
files += [x['archived_source'] for x in dependencies]
files += [str(x.relative_to(ROOT)) for x in (ROOT/'ops').iterdir() if x.is_file() and x.suffix in {'.py','.md','.json','.txt'} and not x.name.startswith('ROUND00_')]
files += ['ops/ROUND00_ALLOWLIST.txt', 'ops/ROUND00_PREPARATION_AUDIT.json']
files = sorted(set(files))
atomic_json(ROOT/'ops/ROUND00_PREPARATION_AUDIT.json', {'prepared_at_utc': stamp(),
    'frozen_manifest_sha256': expected_manifest, 'independent_review_passed': True,
    'dependency_copies_match_frozen_source_hashes': True, 'copied_dependency_count': len(dependencies),
    'source_and_settings_frozen_before_fit': True, 'trained_new_pilot_jobs': False,
    'download_required_before_git_staging': True, 'excluded_server_only_artifacts': ['baselines/calibrated_eta0.pt','raw datasets','prediction arrays','__pycache__'],
    'initial_manual_health_snapshot_superseded': 'monitoring/CHECK_20260929T155302Z.json; see ops/README.md',
    'git_transport': 'Direct SSH22/SSH443/HTTPS443 failed. Scoped HTTPS proxy from enabled Windows system settings127.0.0.1:7890 succeeded; existing credential helper only. No global config changes.',
    'github_read_verified_main': '69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e'})
(ROOT/'ops/ROUND00_ALLOWLIST.txt').write_text('\n'.join(files)+'\n')
entries = inspect(ROOT, files)
print(json.dumps({'files':len(entries),'bytes':sum(x['bytes'] for x in entries),'paths':[x['path'] for x in entries]},indent=2))
