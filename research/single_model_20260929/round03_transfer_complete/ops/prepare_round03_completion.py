#!/usr/bin/env python3
"""Prospective Round03 records only; no job polling or completed-round packaging."""
import hashlib
import json
import re
from pathlib import Path
from monitor import ROOT, atomic_json, safe_path, stamp

MANIFEST = '93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37'
PACKAGES = {
    'ao_integral_feasibility/MANIFEST.json': '82e09937178c5474db09cd5b2890189fb4d4e857f887cc58262a474ac8059172',
    'reports/NEXT_DIRECTIONS_MANIFEST.json': '818131c5cc4c5439f3d54bab7feda6710df44b7b1875b0c211880fb79f488557',
}
EXTRA_HASHES = {
    'reports/CALIBRATION_HISTORY_CHECK.md': '0e3c25ae3080ca547de9330075006b67836a76fd40d2fd81b3d3361f441e3034',
    'round03_transfer/TERMINAL_REVIEW_CHECKLIST.md': '16d3d1b0859037474ee8a5ffb2c79e3e9804e1f79b1cfff5b1d64f5808980ee1',
    'reports/NEXT_DIRECTION_FEATURE_CLARIFICATION.md': '711eebede513d57b024012a65f9f6ec1bc3b5c7be9071140fe686f3827c6601e',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rd = ROOT / 'round03_transfer'
    assert digest(rd / 'FROZEN_MANIFEST.json') == MANIFEST
    groups = {
        'published_preparation_records': (ROOT / 'ops/ROUND03_PREPARATION_ALLOWLIST.txt').read_text().splitlines(),
        'source_terminal_records': ['round03_transfer/runs/source33/' + n for n in (
            'LAUNCH_RECEIPT.json', 'MONITOR_REGISTRATION.json', 'TRAIN_DATA_ACCESS.json',
            'history.jsonl', 'train.log', 'status.json', 'FIT_COMPLETE.json')],
        'affine_terminal_records': ['round03_transfer/affine/' + n for n in (
            'COEFFICIENTS_FROZEN.json', 'EXPORT_COMPLETE.json', 'VALIDATION_COMPLETE.json')],
        'affine_attempt_evidence': [],
        'operational_records': [
            'ops/prepare_round03_completion.py', 'ops/ROUND03_COMPLETION_ARCHIVE_NOTES.md',
            'ops/check_round03_startup.py', 'ops/ROUND03_STARTUP_CHECK.json',
            'ops/check_round03_continuation.py', 'monitoring/CHECK_20260929T185538Z.json',
            'monitoring/ROUND03_HEARTBEAT_20260929T194927.json',
            'monitoring/ROUND03_HEARTBEAT_20260929T195916.json',
            'ops/ROUND03_TERMINAL_RESOURCE_OBSERVATION.json',
            'ops/HEARTBEAT_BEFORE_ROUND03.toml', 'ops/HEARTBEAT_AFTER_ROUND03.toml',
            'ops/ROUND03_HEARTBEAT_VERIFICATION.json', 'ops/ROUND03_RUNTIME_HANDOFF.md',
            'ops/ROUND03_PUBLICATION_RECEIPT.json', 'current_state/science_implementation.md',
            'current_state/COORDINATOR.md', 'current_state/monitor.md'],
        'preparation_publication_provenance': ['ops/publication_records/round03_preparation/' + n for n in (
            'LOCAL_DOWNLOAD_RECEIPT.json', 'STAGED_REVIEW.json', 'CONTENT_REVIEW.json',
            'STAGED_DIFF.patch', 'LOCAL_COMMIT_RECEIPT.json', 'PUBLICATION_RECEIPT.json',
            'ROUND03_PUBLICATION_RECEIPT.json', 'COMMIT_MESSAGE.txt')],
        'supplements': list(EXTRA_HASHES),
    }
    for name, expected in EXTRA_HASHES.items():
        assert digest(safe_path(ROOT, name)) == expected, name
    for name, expected in PACKAGES.items():
        path = safe_path(ROOT, name)
        assert digest(path) == expected, name
        groups['supplements'].append(name)
        for member, meta in json.loads(path.read_text())['files'].items():
            relative = (path.parent / member).relative_to(ROOT).as_posix()
            target = safe_path(ROOT, relative)
            assert digest(target) == meta['sha256'], relative
            assert target.stat().st_size == meta['bytes'], relative
            groups['supplements'].append(relative)
    # Names only: inventory does not inspect current job status, logs or processes.
    for path in rd.iterdir():
        if path.is_file() and re.fullmatch(r'AFFINE_(?:FIT|EVALUATE)_GPU_ADMISSION\.xml|AFFINE_(?:FIT|VALIDATION)_EXECUTION_[A-Za-z0-9_-]+\.log', path.name):
            groups['affine_attempt_evidence'].append(path.relative_to(ROOT).as_posix())
    if (rd / 'affine').exists():
        for path in (rd / 'affine').iterdir():
            if path.is_file() and re.fullmatch(r'VALIDATION_ATTEMPT_\d+\.json', path.name):
                groups['affine_attempt_evidence'].append(path.relative_to(ROOT).as_posix())
    names = sorted(set(name for values in groups.values() for name in values if name))
    inventory = {
        'prepared_at_utc': stamp(), 'status': 'prospective_only_not_a_completed_round',
        'source_manifest_sha256': MANIFEST, 'groups': groups,
        'missing_expected_files': [name for name in names if not safe_path(ROOT, name).is_file()],
        'supplement_manifest_hashes': PACKAGES, 'supplement_record_hashes': EXTRA_HASHES,
        'pending_additions': ['actual source-exit/checkpoint-hash receipt', 'final science aggregate summary and figures',
            'independent bound terminal scientific review and report manifest', 'root Round03 decision',
            'final current handoffs', 'preserved failure/interruption receipts, if any'],
        'successful_round_gates': ['source33 exact terminal/manifest/exit/final+last hashes',
            'immutable coefficients and verified single-checkpoint exports', 'single completed fixed validation',
            'science aggregate analysis', 'independent review PASS', 'root decision'],
        'next_parent': 'remotely verified current research-branch descendant of 1b34e839335177e1fd582944549be4f181dee3c0',
        'required_order': ['finalize explicit lightweight allowlist', 'package server records',
            'FIRST download to new D archive and verify all bytes', 'stage --no-filters and inspect every blob',
            'commit and non-force push', 'verify remote receipt'],
        'failure_policy': 'ops/FAILED_ROUND_ARCHIVE_POLICY.md; archive honest failure only after root declaration',
        'process_polling_performed': False, 'bundle_created': False, 'completion_claimed': False,
        'checkpoint_data_prediction_cache_or_credential_artifacts_included': False,
    }
    atomic_json(ROOT / 'ops/ROUND03_COMPLETION_PROSPECTIVE_INVENTORY.json', inventory)
    print(json.dumps({'status': inventory['status'], 'known_records': len(names),
        'missing_expected_files': inventory['missing_expected_files'], 'supplement_hashes_verified': True,
        'bundle_created': False, 'process_polling_performed': False}, indent=2))

if __name__ == '__main__':
    main()
