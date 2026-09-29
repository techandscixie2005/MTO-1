#!/usr/bin/env python3
"""Finalize an exact lightweight Round03 archive after terminal review and decision."""
import argparse
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, file_info, process_identity, safe_path, stamp
from package_records import inspect
from prepare_round03_completion import MANIFEST, PACKAGES, EXTRA_HASHES

PREFIX = 'round03_transfer/'
FINAL_RECORDS = (
    'verify_source_gate.py', 'TERMINAL_SOURCE_GATE.json', 'run_validation_bound.py',
    'RESOURCE_PLACEMENT_NOTE.md', 'GPU_PLACEMENT_REVIEW.md', 'summarize_transfer.py',
    'ROUND03_RESULTS.json', 'ROUND03_RESULTS.md', 'ANALYSIS_RECEIPT.json',
    'SUMMARY_EXECUTION.log', 'ROUND03_REPORT.md', 'ROUND03_REPRODUCE.md',
    'AFFINE_FIT_EXECUTION_20260929T2001252036053Z.log', 'AFFINE_FIT_GPU_ADMISSION.xml',
    'AFFINE_VALIDATION_EXECUTION_20260929T200405597244Z.log', 'AFFINE_EVALUATE_GPU_ADMISSION.xml',
    'VALIDATION_LAUNCH_20260929T200405597244Z.json',
    'affine/VALIDATION_ATTEMPT_1790712249756330193.json',
    'independent_terminal_review.py', 'INDEPENDENT_TERMINAL_INTEGRITY.json',
    'INDEPENDENT_SCIENTIFIC_REVIEW.md', 'INDEPENDENT_SCIENTIFIC_REVIEW.json',
    'SINGLE_MODEL_RESEARCH_REPORT.md', 'INDEPENDENT_REVIEW_MANIFEST.json',
    'runs/source33/round03_source33_1790707949921845597_gpu_admission.xml',
)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-manifest-sha256', required=True)
    parser.add_argument('--decision-sha256', required=True)
    args = parser.parse_args()
    rd = ROOT / 'round03_transfer'
    assert sha(rd / 'FROZEN_MANIFEST.json') == MANIFEST
    review_manifest = rd / 'INDEPENDENT_REVIEW_MANIFEST.json'
    assert sha(review_manifest) == args.review_manifest_sha256
    for filename, expected in read(review_manifest)['files'].items():
        assert sha(Path(filename)) == expected, filename
    decision = ROOT / 'current_state/ROUND03_DECISION.md'
    assert sha(decision) == args.decision_sha256 and len(decision.read_text()) > 100
    scientific = read(rd / 'INDEPENDENT_SCIENTIFIC_REVIEW.json')
    integrity = read(rd / 'INDEPENDENT_TERMINAL_INTEGRITY.json')
    analysis = read(rd / 'ANALYSIS_RECEIPT.json')
    for record in (scientific, integrity, analysis):
        assert record['passed']
        assert MANIFEST in (record.get('manifest_sha256'), record.get('frozen_manifest_sha256'))
    assert not scientific['unresolved_integrity_findings'] and scientific['all_valid_labels_preserved']
    assert scientific['resource_placement_caveat_preserved']
    assert not integrity['completed_affine_fit_gpu_placement_verified']
    assert integrity['evaluation_gpu1_placement_observed']
    assert sha(rd / 'INDEPENDENT_TERMINAL_INTEGRITY.json') == scientific['integrity_review_sha256']
    assert sha(rd / 'INDEPENDENT_SCIENTIFIC_REVIEW.md') == scientific['scientific_review_sha256']
    assert sha(rd / 'ANALYSIS_RECEIPT.json') == integrity['analysis_receipt_sha256']
    assert sha(rd / 'ROUND03_RESULTS.json') == scientific['results_sha256']
    for category in ('source_hashes', 'input_hashes', 'output_hashes'):
        for filename, expected in analysis[category].items():
            assert sha(Path(filename)) == expected, filename
    source = rd / 'runs/source33'
    complete = read(source / 'FIT_COMPLETE.json')
    assert complete['completed_epoch'] == 33 and complete['steps'] == 49665
    assert complete['fixed_epoch_selection'] and complete['manifest_sha256'] == MANIFEST
    assert not complete['heldout_or_outer_labels_used_during_fit'] and not complete['test_evaluated']
    assert not (source / 'FAILED.json').exists()
    history = [json.loads(line) for line in (source / 'history.jsonl').read_text().splitlines() if line.strip()]
    assert [row['epoch'] for row in history] == list(range(1, 34))
    assert all('validation' not in row for row in history)
    registry = read(ROOT / 'ops/registry.json')
    attempts = [run for run in registry['runs'] if run['run_dir'] == PREFIX + 'runs/source33']
    assert attempts
    for run in attempts:
        current = process_identity(run['identity']['pid'])
        if current:
            same = all(current[k] == run['identity'][k] for k in ('pid', 'start_ticks', 'boot_id', 'uid', 'cwd', 'argv_sha256'))
            assert not same or current['process_state'] == 'Z', 'Owned source still live'
    checkpoint_metadata = {}
    for name, key in [('source_final.pt', 'checkpoint_sha256'), ('last.pt', 'last_checkpoint_sha256')]:
        expected = complete[key]
        assert sha(source / name) == expected
        checkpoint_metadata[name] = {**file_info(source / name), 'sha256': expected, 'server_only': True}
    affine = rd / 'affine'
    coefficients = read(affine / 'COEFFICIENTS_FROZEN.json')
    exports = read(affine / 'EXPORT_COMPLETE.json')
    validation = read(affine / 'VALIDATION_COMPLETE.json')
    assert all(r['manifest_sha256'] == MANIFEST for r in (coefficients, exports, validation))
    assert not coefficients['outer_validation_evaluated_before_freeze'] and not coefficients['test_evaluated']
    assert exports['single_model_single_checkpoint']
    assert not validation['test_evaluated'] and not validation['prediction_averaging']
    assert validation['validation_reused_historically'] and validation['validation_molecules'] == 6686
    assert all(m['raw_f']['count'] == 66860 for m in validation['metrics'].values())
    assert sha(affine / 'COEFFICIENTS_FROZEN.json') == exports['coefficient_receipt_sha256'] == validation['coefficient_receipt_sha256']
    assert sha(affine / 'EXPORT_COMPLETE.json') == validation['export_receipt_sha256']
    assert len(list(affine.glob('VALIDATION_ATTEMPT_*.json'))) == 1
    for name, record in exports['checkpoints'].items():
        assert sha(Path(record['path'])) == record['sha256']
        checkpoint_metadata[name] = {**record, **file_info(Path(record['path'])), 'server_only': True}
    launch = read(rd / 'VALIDATION_LAUNCH_20260929T200405597244Z.json')
    assert launch['exit_code'] == 0 and launch['uuid_mask_set_before_interpreter']
    assert launch['actual_gpu1_placement_observed'] and launch['pinned_affine_artifacts_unchanged']
    assert launch['observed_gpu_uuids'] == ['GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800']
    assert sha(affine / 'VALIDATION_COMPLETE.json') == launch['validation_receipt_sha256']
    prospective = read(ROOT / 'ops/ROUND03_COMPLETION_PROSPECTIVE_INVENTORY.json')
    names = {n for group in prospective['groups'].values() for n in group if n}
    names.update(Path(filename).relative_to(ROOT).as_posix() for filename in read(review_manifest)['files'])
    names.update(PREFIX + n for n in FINAL_RECORDS)
    names.update(('ops/finalize_round03_completion.py', 'ops/prepare_round03_completion.py',
        'ops/ROUND03_COMPLETION_PROSPECTIVE_INVENTORY.json', 'ops/ROUND03_TERMINAL_RESOURCE_OBSERVATION.json',
        'monitoring/ROUND03_HEARTBEAT_20260929T195916.json', 'current_state/ROUND03_DECISION.md',
        'current_state/COORDINATOR.md', 'current_state/monitor.md', 'history_baseline.md'))
    for name, expected in {**PACKAGES, **EXTRA_HASHES}.items():
        assert sha(safe_path(ROOT, name)) == expected, name
    # Preserve any incident records explicitly instead of dropping them from a successful archive.
    incident_names = []
    for directory in (rd, source, affine):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in {'.json', '.log', '.md'} and any(t in path.name for t in ('FAILED', 'RESUMED_FAILURE', 'INITIAL_', 'INCIDENT')):
                incident_names.append(path.relative_to(ROOT).as_posix())
    names.update(incident_names)
    records = inspect(ROOT, sorted(names))
    final = {'finalized_at_utc': stamp(), 'status': 'completed_round_reviewed_ready_to_package',
        'source_manifest_sha256': MANIFEST, 'independent_review_manifest_sha256': args.review_manifest_sha256,
        'root_decision_sha256': args.decision_sha256, 'scientific_review_passed': True,
        'all_valid_labels_preserved': True, 'test_evaluated': False, 'validation_reused_historically': True,
        'incumbent_retained': True, 'completed_affine_fit_gpu_placement_verified': False,
        'validation_gpu1_placement_observed': True, 'checkpoint_metadata_only': checkpoint_metadata,
        'preserved_incident_paths': incident_names, 'file_count_before_inventory_and_allowlist': len(records),
        'files': records, 'next_parent': 'verified research branch descendant of1b34e839; no force or rebase',
        'archive_must_precede_stage_commit_push': True, 'checkpoints_arrays_data_caches_credentials_exported': False}
    atomic_json(ROOT / 'ops/ROUND03_COMPLETION_INVENTORY.json', final)
    names.update(('ops/ROUND03_COMPLETION_INVENTORY.json', 'ops/ROUND03_COMPLETION_ALLOWLIST.txt'))
    (ROOT / 'ops/ROUND03_COMPLETION_ALLOWLIST.txt').write_text('\n'.join(sorted(names)) + '\n')
    records = inspect(ROOT, sorted(names))
    print(json.dumps({'status': final['status'], 'files': len(records), 'bytes': sum(r['bytes'] for r in records),
        'manifest_sha256': MANIFEST, 'review_manifest_sha256': args.review_manifest_sha256,
        'decision_sha256': args.decision_sha256, 'fit_placement_caveat_preserved': True}))

if __name__ == '__main__':
    main()
