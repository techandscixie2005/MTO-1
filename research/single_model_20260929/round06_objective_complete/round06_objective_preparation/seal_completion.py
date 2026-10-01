"""Seal explicit lightweight Round06 records; never copy private arrays/tensors."""
import json
import time
from common import ROOT, sha, read, atomic_json, require_cpu

require_cpu()
output = ROOT / 'completion/TERMINAL_MANIFEST.json'
assert not output.exists(), 'Completion manifest is immutable'
receipt = read(ROOT / 'completion/ANALYSIS_RECEIPT.json')
assert receipt['passed'] and receipt['no_model_inference']
manifest_sha = sha(ROOT / 'FROZEN_MANIFEST.json')
assert manifest_sha == '4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b'
assert receipt['manifest_sha256'] == manifest_sha
names = [
    'terminal_analysis.py', 'TERMINAL_ANALYSIS_CONFIG.json',
    'TERMINAL_ANALYSIS_SOURCE_REVIEW.json', 'write_completion_report.py',
    'seal_completion.py', 'inspect_first_epoch.py',
    'FIRST_EPOCH_INSPECTOR_SOURCE_REVIEW.json', 'FIRST_EPOCH_INSPECTION_NOTE.md',
    'PRODUCTION_EXECUTION_AUTHORIZATION.json', 'CURRENT_RUNTIME_HANDOFF.md',
    'ops/FIRST_EPOCH_METADATA.json', 'ops/FIRST_EPOCH_INSPECTION.log',
    'ops/terminal_analysis_01.log', 'ops/write_completion_report_01.log',
    'completion/ROUND06_RESULTS.json', 'completion/ANALYSIS_RECEIPT.json',
    'completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg',
    'completion/ROUND06_REPORT.md', 'completion/ROUND06_REPRODUCE.md',
    'completion/DEFERRED_CONGRUENCE_ASSESSMENT.md',
    'completion/CONTROL_REPEAT_CONTEXT.md',
]
paths = [ROOT / name for name in names]
for arm in ('trace_control', 'raw_f'):
    run = ROOT / 'runs' / arm
    terminal = read(run / 'FIT_COMPLETE.json')
    assert terminal['completed_epoch'] == 60 and terminal['steps'] == 112860
    assert terminal['manifest_sha256'] == manifest_sha
    assert terminal['test_evaluated'] is False and not (run / 'FAILED.json').exists()
    paths += [run / name for name in
              ('FIT_COMPLETE.json', 'BEST.json', 'DATA_ACCESS.json', 'history.jsonl', 'status.json')]
    attempts = list((run / 'attempts').iterdir())
    assert len(attempts) == 1 and attempts[0].is_dir()
    paths += [attempts[0] / name for name in
              ('LAUNCH_RECEIPT.json', 'GPU_ADMISSION.xml', 'REGISTRATION_TOOL.log', 'train.log')]
assert len(paths) == len(set(paths))
entries = []
for path in paths:
    assert path.is_file() and not path.is_symlink()
    assert path.suffix.lower() in ('.json', '.jsonl', '.md', '.py', '.xml', '.log', '.svg')
    path.read_text(encoding='utf-8')
    entries.append({'path': str(path), 'relative_path': str(path.relative_to(ROOT)),
                    'sha256': sha(path), 'bytes': path.stat().st_size})
atomic_json({
    'format': 'round06_lightweight_terminal_manifest_v1', 'created_unix': time.time(),
    'frozen_scientific_manifest_sha256': manifest_sha,
    'preparation_commit': 'c26a860b972a1c01267f8b766393772173515b92',
    'analysis_receipt_sha256': sha(ROOT / 'completion/ANALYSIS_RECEIPT.json'),
    'count': len(entries), 'bytes': sum(entry['bytes'] for entry in entries), 'files': entries,
    'files_are_exact_archive_payload_allowlist': True,
    'private_input_hashes_are_provenance_only': True,
    'inherited_sources': 'Exact frozen134-file source closure and prior failure/preflight/split records are in the preparation publication; restore by its manifest and dependency mapping.',
    'independent_reviews_root_decision_and_current_monitor_records': 'Publisher adds exact reviewed records separately; no circular binding.',
    'exclusions': ['all checkpoint and optimizer tensors', 'raw or prediction arrays',
                   'split or identity arrays', 'raw datasets', 'caches', 'credentials',
                   'private preflight directories'],
    'test_evaluated': False, 'completed_two_arm_fixed60': True,
}, output)
print(json.dumps({'manifest_sha256': sha(output), 'count': len(entries),
                  'bytes': sum(entry['bytes'] for entry in entries)}))
