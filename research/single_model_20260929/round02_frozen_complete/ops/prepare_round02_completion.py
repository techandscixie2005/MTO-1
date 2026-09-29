#!/usr/bin/env python3
"""Explicit Round02 completion inventory; never publish partial results as complete."""
import argparse
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, file_info, process_identity, safe_path, stamp
from package_records import inspect

EXPECTED = '6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e'
PREFIX = 'round02_frozen/'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--review')
    p.add_argument('--decision')
    p.add_argument('--extra', action='append', default=[])
    p.add_argument('--package-manifest', action='append', default=[])
    a = p.parse_args()
    rd = ROOT / 'round02_frozen'
    assert digest(rd / 'FROZEN_MANIFEST.json') == EXPECTED
    groups = {
        'source_and_preflight': [s for s in (ROOT / 'ops/ROUND02_PREPARATION_ALLOWLIST.txt').read_text().splitlines() if s],
        'terminal_analysis': [PREFIX + n for n in (
            'terminal_replay.py', 'TERMINAL_REPLAY.json', 'TERMINAL_REPLAY_EXECUTION.log',
            'TERMINAL_REPLAY_GPU_ADMISSION.xml', 'summarize_frozen.py', 'ROUND02_RESULTS.json',
            'ROUND02_RESULTS.md', 'ANALYSIS_RECEIPT.json', 'SUMMARY_EXECUTION.log',
            'plot_frozen.py', 'ROUND02_CURVES.svg', 'PLOT_MANIFEST.json', 'PLOT_EXECUTION.log',
            'checkpoint_inventory.py', 'CHECKPOINT_INVENTORY.json', 'CHECKPOINT_INVENTORY_EXECUTION.log',
            'TERMINAL_REPLAY_INITIAL_IMPORT_CHECK.log', 'train_calibration_audit.py',
            'TRAIN_CALIBRATION_AUDIT.json', 'TRAIN_CALIBRATION_EXECUTION.log',
            'TRAIN_CALIBRATION_GPU_ADMISSION.xml', 'ROUND02_INTERPRETATION.md')],
        'arm_records': [], 'supplements': [], 'review_and_decision': [],
        'ops_and_monitoring': ['ops/prepare_round02_completion.py', 'ops/check_round02_startup.py',
            'ops/ROUND02_STARTUP_CHECK.json', 'ops/ROUND02_PUBLICATION_RECEIPT.json',
            'ops/ROUND02_COMPLETION_ARCHIVE_NOTES.md', 'ops/test_archive_safety.py',
            'ops/ROUND02_ARCHIVE_SAFETY_TEST.json', 'summarize.py'],
    }
    # Prospective mode inventories known filenames only: no process or job-state poll.
    for arm in ('trace', 'raw_f'):
        out = rd / 'runs' / arm
        groups['arm_records'] += [PREFIX + 'runs/' + arm + '/' + n for n in (
            'LAUNCH_RECEIPT.json', 'MONITOR_REGISTRATION.json', 'history.jsonl', 'train.log',
            'status.json', 'best_metrics.json', 'FIT_COMPLETE.json')]
        if out.exists():
            groups['arm_records'] += [x.relative_to(ROOT).as_posix() for x in out.iterdir()
                if x.is_file() and (x.name.endswith('_gpu_admission.xml') or
                    (('FAILED' in x.name or 'previous_failure' in x.name) and x.suffix in {'.json', '.log'}))]
    package_manifests = ['atom_covariance_feasibility/MANIFEST.json',
                         'optimizer_state_audit/MANIFEST.json', *a.package_manifest]
    for name in package_manifests:
        path = safe_path(ROOT, name)
        groups['supplements'].append(name)
        if not path.exists():
            continue
        package = json.loads(path.read_text())
        for member, metadata in package['files'].items():
            target = safe_path(ROOT, (path.parent / member).relative_to(ROOT).as_posix())
            assert digest(target) == metadata['sha256'], str(target)
            groups['supplements'].append(target.relative_to(ROOT).as_posix())
    for name in (a.review, a.decision, *a.extra):
        if name:
            safe_path(ROOT, name)
            groups['review_and_decision'].append(name)
    for folder in ('monitoring', 'ops/publication_records/round02_preparation'):
        directory = ROOT / folder
        if directory.exists():
            groups['ops_and_monitoring'] += [x.relative_to(ROOT).as_posix() for x in directory.iterdir()
                if x.is_file() and x.suffix in {'.json', '.jsonl', '.log'}]
    terminal = {}
    if a.finalize:
        assert a.review and a.decision, 'Independent PASS and root decision required'
        review = json.loads(safe_path(ROOT, a.review).read_text())
        assert review.get('passed') is True
        assert EXPECTED in (review.get('manifest_sha256'), review.get('frozen_manifest_sha256'),
                            review.get('source_manifest_sha256')), 'Review must bind exact Round02 source'
        assert len(safe_path(ROOT, a.decision).read_text().strip()) > 100
        final_report_manifest = json.loads((rd / 'INDEPENDENT_REVIEW_MANIFEST.json').read_text())
        assert digest(rd / 'INDEPENDENT_REVIEW_MANIFEST.json') == '2d8ccbe0591bd1f5a2f3cd0520ff756ab9d1e104b5589880fc183b19e08fe03a'
        assert final_report_manifest['root_next_decision_pending'] is False
        assert digest(safe_path(ROOT, a.decision)) == final_report_manifest['root_decision']['sha256']
        train_review = json.loads((rd / 'TRAIN_CALIBRATION_REVIEW.json').read_text())
        assert train_review['passed']
        assert digest(rd / 'TRAIN_CALIBRATION_AUDIT.json') == train_review['input_sha256']
        assert digest(rd / 'review_train_moments.py') == train_review['script_sha256']
        result = json.loads((rd / 'ROUND02_RESULTS.json').read_text())
        assert result['passed'] and result['manifest_sha256'] == EXPECTED
        assert result['single_checkpoint_per_arm'] and result['predictions_averaged'] is False
        assert result['test_evaluated'] is False
        assert set(result['arms']) == {'trace', 'raw_f'}
        assert digest(rd / 'ANALYSIS_RECEIPT.json') == review['analysis_receipt_sha256']
        assert digest(rd / 'TERMINAL_REPLAY.json') == review['terminal_replay_sha256']
        analysis = json.loads((rd / 'ANALYSIS_RECEIPT.json').read_text())
        assert analysis['passed'] and analysis['manifest_sha256'] == EXPECTED
        for name, expected in analysis['output_hashes'].items():
            assert digest(safe_path(ROOT, PREFIX + name)) == expected, name
        for path, expected in analysis['source_hashes'].items():
            assert digest(Path(path)) == expected, path
        replay = json.loads((rd / 'TERMINAL_REPLAY.json').read_text())
        assert replay['passed'] and replay['manifest_sha256'] == EXPECTED
        assert replay['raw_arrays_written'] is False and replay['test_inference'] is False
        registry = json.loads((ROOT / 'ops/registry.json').read_text())
        for arm in ('trace', 'raw_f'):
            out = rd / 'runs' / arm
            c = json.loads((out / 'FIT_COMPLETE.json').read_text())
            assert c['completed_epoch'] == 20 and c['single_checkpoint']
            assert not c['test_evaluated'] and c['geometry_only_inference']
            assert c['frozen_parameters_and_all_buffers_unchanged']
            assert c['manifest_sha256'] == EXPECTED and not (out / 'FAILED.json').exists()
            for name in ('FIT_COMPLETE.json', 'history.jsonl'):
                assert digest(out / name) == review['input_hashes'][arm][name]
            assert result['arms'][arm]['selected_epoch'] == c['selected_epoch']
            history = [json.loads(s) for s in (out / 'history.jsonl').read_text().splitlines() if s]
            assert [r['epoch'] for r in history] == list(range(21))
            assert min(history, key=lambda r: r['validation']['raw_f']['sse'])['epoch'] == c['selected_epoch']
            owned = [r for r in registry['runs'] if r['run_dir'] == PREFIX + 'runs/' + arm]
            assert owned
            for run in owned:
                observed = process_identity(run['identity']['pid'])
                if observed:
                    exact = all(observed[k] == run['identity'][k] for k in ('pid', 'start_ticks', 'boot_id', 'uid', 'cwd', 'argv_sha256'))
                    assert not exact or observed['process_state'] == 'Z', 'Exact owned training worker still live'
            terminal[arm] = {'completed_epoch': 20, 'selected_epoch': c['selected_epoch'],
                'checkpoint_metadata_only': {n: {**file_info(out / n), 'sha256_from_terminal_receipt': c[tag]}
                    for n, tag in [('best.pt', 'best_checkpoint_sha256'), ('last.pt', 'last_checkpoint_sha256')]},
                'single_checkpoint': True, 'test_evaluated': False}
    names = sorted(set(n for values in groups.values() for n in values))
    missing = [n for n in names if not (ROOT / n).is_file()]
    inventory = {'prepared_at_utc': stamp(),
        'status': 'ready_to_package' if a.finalize else 'prospective_only_not_a_completed_round',
        'frozen_manifest_sha256': EXPECTED, 'groups': groups, 'missing_files': missing,
        'terminal_states_checked_only_at_finalization': terminal,
        'published_preparation_commit': 'a4c2fabd837bea5ae7eced79d11de797af3900bf',
        'next_commit_parent': 'remotely verified current research branch descendant; no force or rebase',
        'sequence': ['both20epoch terminal receipts and selected/last replay', 'independent bound review PASS and root decision',
            'mirror local supplemental records', 'explicit allowlist content/size inspection',
            'package then download FIRST to new D archive', 'verify every hash',
            'stage --no-filters and byte-review', 'commit and non-force push', 'verify remote commit'],
        'failed_or_aborted_rounds': 'Separate honest incident/failure bundle under FAILED_ROUND_ARCHIVE_POLICY.md; no fabricated completion',
        'weights_optimizer_data_cache_and_prediction_arrays_exported': False,
        'frozen_source_or_previous_archive_rewritten': False}
    destination = ROOT / 'ops/ROUND02_COMPLETION_INVENTORY.json'
    atomic_json(destination, inventory)
    if a.finalize:
        assert not missing, missing
        names += ['ops/ROUND02_COMPLETION_INVENTORY.json', 'ops/ROUND02_COMPLETION_ALLOWLIST.txt']
        (ROOT / 'ops/ROUND02_COMPLETION_ALLOWLIST.txt').write_text('\n'.join(sorted(set(names))) + '\n')
        files = inspect(ROOT, names)
        print(json.dumps({'status': 'ready_to_package', 'files': len(files), 'bytes': sum(x['bytes'] for x in files)}))
    else:
        print(json.dumps({'status': inventory['status'], 'known_files': len(names), 'missing_files': missing}))

if __name__ == '__main__':
    main()
