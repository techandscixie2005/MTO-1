#!/usr/bin/env python3
"""Prepare a bounded completion allowlist; no scientific execution or private arrays."""
import argparse
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, stamp
from package_records import inspect

RD = ROOT / 'round04_fonly_preparation'
PARENT = '74810c5ad8a984fbf119dccd2e22336a112de7f5'
SOURCE = '79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f'
OP = ROOT / 'ops/main_publication'
p = argparse.ArgumentParser()
p.add_argument('--review-json', required=True)
p.add_argument('--review-md', required=True)
p.add_argument('--review-sha256', required=True)
p.add_argument('--review-manifest-sha256', required=True)
p.add_argument('--decision', required=True)
p.add_argument('--decision-sha256', required=True)
p.add_argument('--readme-sha256', required=True)
a = p.parse_args()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scoped(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT.resolve()) and path.is_file() and not path.is_symlink()
    return path

assert sha(scoped(a.review_json)) == a.review_sha256
review = json.loads(scoped(a.review_json).read_bytes())
assert review['passed'], 'Independent terminal review must pass'
assert sha(scoped(a.decision)) == a.decision_sha256
assert sha(OP / 'README.md') == a.readme_sha256
assert sha(RD / 'FROZEN_MANIFEST.json') == SOURCE
assert not (OP / 'COMPLETION_ARCHIVE_INVENTORY.json').exists(), 'Preserve previous inventory'
manifest = json.loads((RD / 'ROUND04_TERMINAL_MANIFEST.json').read_bytes())
assert review['source_manifest_sha256'] == SOURCE
assert review['terminal_manifest_sha256'] == sha(RD / 'ROUND04_TERMINAL_MANIFEST.json')
assert manifest['phase'] == 'completed_round04' and manifest['aggregate_arithmetic_passed']
assert manifest['no_new_test_or_historical_test_access']
names = set()
review_manifest_path = RD / 'INDEPENDENT_TERMINAL_REVIEW_MANIFEST.json'
assert sha(review_manifest_path) == a.review_manifest_sha256
review_manifest = json.loads(review_manifest_path.read_bytes())
assert review_manifest['passed']
for name, expected in review_manifest['files'].items():
    path = (RD / name).resolve()
    assert path.is_relative_to(RD.resolve()) and sha(path) == expected, name
    names.add(path.relative_to(ROOT).as_posix())
for name, expected in manifest['input_hashes'].items():
    path = (RD / name).resolve()
    assert path.is_relative_to(RD.resolve()) and sha(path) == expected, name
    names.add(path.relative_to(ROOT).as_posix())
names.update((
    'round04_fonly_preparation/ROUND04_TERMINAL_MANIFEST.json', a.review_json, a.review_md, a.decision,
    'round04_fonly_preparation/independent_terminal_review.py',
    'round04_fonly_preparation/INDEPENDENT_TERMINAL_INTEGRITY.json',
    'round04_fonly_preparation/INDEPENDENT_TERMINAL_REVIEW_MANIFEST.json',
    'round04_fonly_preparation/ops/ARCHIVE_INVENTORY.json',
    'round04_fonly_preparation/ops/INHERITED_DEPENDENCY_VERIFICATION.json',
    'current_state/COORDINATOR.md', 'current_state/monitor.md',
    'current_state/science_implementation.md', 'current_state/RESEARCH_RESUMPTION_20260930.md',
    'ops/ROUND04_PREPARATION_PUBLICATION_RECEIPT.json', 'ops/register_cpu_stage.py',
    'ops/monitor.py', 'ops/registry.json', 'ops/package_records.py', 'ops/download_records.py',
    'ops/publish_main_results.py', 'ops/prepare_round04_completion.py', 'ops/ssh_proxy.py',
    'ops/FAILED_ROUND_ARCHIVE_POLICY.md', 'ops/main_publication/README.md',
))
restore = OP / 'COMPLETION_RESTORE.md'
assert not restore.exists()
restore.write_text('# Round04 completion restoration\n\n'
    'This completion archive supplements the immutable preparation archive at commit '+PARENT+'. '
    'Restore scientific sources from research/single_model_20260929/round04_fonly_preparation/ '
    'using its ops/ARCHIVE_INVENTORY.json mapping and the original absolute server layout. '
    'The source manifest remains '+SOURCE+'. '
    'The two inherited Round03 text dependencies are mapped by '
    'round04_fonly_preparation/ops/INHERITED_DEPENDENCY_VERIFICATION.json. '
    'The new completion report and wrapper document the actual authorized execution. '
    'Completion markers block rerunning completed stages.\n\n'
    'All source/config/log/aggregate records needed to understand the result are in this archive '
    'or that verified ancestor. Checkpoint tensors, learned coefficient tensors, raw labels, '
    'identity/split membership arrays, prediction caches and credentials remain private server artifacts. '
    'Their hashes and server paths are provenance only. Unfinished dataset-builder/QC drafts are excluded. '
    'The README snapshot is staged at repository README.md; other members are archived under '
    'research/single_model_20260929/round04_fonly_complete/.\n', encoding='utf-8')
names.add(restore.relative_to(ROOT).as_posix())
files = inspect(ROOT, sorted(names))
atomic_json(OP / 'COMPLETION_ARCHIVE_INVENTORY.json', dict(prepared_at_utc=stamp(),
    phase='completed_round04_reviewed_for_publication', parent=PARENT,
    source_manifest_sha256=SOURCE, independent_terminal_review_sha256=a.review_sha256,
    independent_terminal_review_manifest_sha256=a.review_manifest_sha256,
    root_decision_sha256=a.decision_sha256, approved_readme_sha256=a.readme_sha256,
    terminal_manifest_sha256=sha(RD / 'ROUND04_TERMINAL_MANIFEST.json'), files=files,
    no_binary_arrays_weights_predictions_credentials=True, unfinished_research_excluded=True,
    required_order=['server package', 'verified D download', 'separate index byte review',
        'independent staged review and root approval', 'atomic non-force main and research push', 'remote verification']))
names.update(('ops/main_publication/COMPLETION_ARCHIVE_INVENTORY.json',
    'ops/main_publication/COMPLETION_ALLOWLIST.txt'))
(OP / 'COMPLETION_ALLOWLIST.txt').write_text('\n'.join(sorted(names)) + '\n', encoding='utf-8')
files = inspect(ROOT, sorted(names))
print(json.dumps(dict(files=len(files), bytes=sum(f['bytes'] for f in files),
    allowlist=str(OP / 'COMPLETION_ALLOWLIST.txt'), paths=[f['path'] for f in files]), indent=2))
