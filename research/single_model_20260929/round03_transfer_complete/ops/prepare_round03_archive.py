#!/usr/bin/env python3
"""Prepare exact reviewed Round03 sources and metadata, excluding all data arrays."""
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, stamp
from package_records import inspect

EXPECTED = '93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37'
REVIEW = 'a6807aae55c59f254f2048b1d696a754f130075e32ef4763dfa1c94f4fb1c04c'
PREDECESSOR = 'bdc28c351b941a58dae49f833284e13cf5fdc46b'
TOP_LEVEL = '''affine_stage.py clean_source.py config.json evaluation.py fit_normalization.json
freeze_round.py FROZEN_MANIFEST.json INDEPENDENT_PRELAUNCH_REVIEW.json
INDEPENDENT_PRELAUNCH_REVIEW.md INFERENCE_PREFLIGHT_EXECUTION.log INFERENCE_PREFLIGHT.json
inference_preflight.py launch_round.py predictor.py PREFLIGHT_EXECUTION.log
PREFLIGHT_GPU_ADMISSION.xml PREFLIGHT.json preflight.py prepare_split_statistics.py
PROTOCOL.md ROUND03_LAUNCH_AUTHORIZATION.md runtime.py SCIENCE_SPLIT_STATISTICS_REVIEW.md
source_config.json split_manifest.json SPLIT_STATISTICS_HANDOFF.md STATISTICS_AUDIT.json
train_source.py'''.split()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rd = ROOT / 'round03_transfer'
    manifest = json.loads((rd / 'FROZEN_MANIFEST.json').read_text())
    review = json.loads((rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json').read_text())
    assert sha(rd / 'FROZEN_MANIFEST.json') == EXPECTED
    assert sha(rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json') == REVIEW
    assert review['passed'] and review['frozen_manifest_sha256'] == EXPECTED
    assert review['source_hashes'] == manifest['source_hashes']
    assert len(manifest['source_hashes']) == 69
    assert sha(rd / 'INDEPENDENT_PRELAUNCH_REVIEW.md') == review['independent_review_text_sha256']
    assert sha(rd / 'SCIENCE_SPLIT_STATISTICS_REVIEW.md') == '6ecd6994ab991339cebe4e19283fd54e8d816cd1b760674d4888bea390e25b1b'
    assert sha(ROOT / 'current_state/ROUND03_LAUNCH_AUTHORIZATION.md') == sha(rd / 'ROUND03_LAUNCH_AUTHORIZATION.md')
    assert not (rd / 'runs/source33').exists(), 'Production source directory already present'
    prior = json.loads((ROOT / 'ops/ROUND02_COMPLETION_PUBLICATION_RECEIPT.json').read_text())
    assert prior['remote_verified'] and prior['commit'] == PREDECESSOR
    for name, expected in review['evidence_hashes'].items():
        assert sha(rd / name) == expected, name
    names = {'round03_transfer/' + n for n in TOP_LEVEL}
    mapping, excluded = [], []
    for original, expected in manifest['source_hashes'].items():
        source = Path(original)
        if source.suffix not in {'.py', '.json', '.md'} or source.name in {'splits.json', 'identity_audit_v2.json'}:
            excluded.append({'server_path': original, 'sha256': expected,
                             'reason': 'server-only checkpoint, raw data, identity membership or split array; reviewer verified'})
            continue
        content = source.read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, original
        content.decode('utf-8-sig')
        assert b'\x00' not in content and len(content) < 1024 * 1024, original
        if source.is_relative_to(ROOT):
            relative = source.relative_to(ROOT)
        else:
            parts = source.relative_to('/home/inspur/MTO-1').parts
            relative = Path('round03_dependency_sources', *['data_metadata' if p == 'data' else p for p in parts])
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        names.add(relative.as_posix())
        mapping.append({'server_source': original, 'archived_source': relative.as_posix(),
                        'sha256': expected, 'bytes': len(content)})
    names.update(('FROZEN_MANIFEST.json', 'current_state/COORDINATOR.md', 'current_state/monitor.md',
        'current_state/ROUND02_DECISION.md', 'current_state/ROUND03_LAUNCH_AUTHORIZATION.md',
        'ops/ROUND02_COMPLETION_PUBLICATION_RECEIPT.json', 'ops/prepare_round03_archive.py',
        'ops/package_records.py', 'ops/download_records.py', 'ops/publish_records.py',
        'ops/verify_staged_records.py', 'ops/PUBLISH_COMMAND.ps1', 'ops/ssh_proxy.py',
        'ops/monitor.py', 'ops/register_run.py', 'ops/registry.json', 'ops/FAILED_ROUND_ARCHIVE_POLICY.md',
        'ops/ROUND03_ARCHIVE_PREPARATION.json', 'ops/ROUND03_DEPENDENCY_RESTORE.md',
        'ops/ROUND03_PREPARATION_ALLOWLIST.txt', 'ops/ROUND03_RUNTIME_HANDOFF.md'))
    atomic_json(ROOT / 'ops/ROUND03_ARCHIVE_PREPARATION.json', {
        'prepared_at_utc': stamp(), 'frozen_manifest_sha256': EXPECTED,
        'independent_review_sha256': REVIEW, 'independent_review_passed': True,
        'independent_review_kind': 'manual/source review and bounded hash/provenance check; JSON and Markdown are complete receipts',
        'expected_publication_parent': PREDECESSOR, 'source_mapping': mapping,
        'server_only_hash_references': excluded, 'verified_lightweight_source_count': len(mapping),
        'raw_data_identity_membership_checkpoint_or_arrays_copied': False,
        'production_source_launched': False, 'download_required_before_staging': True})
    (ROOT / 'ops/ROUND03_DEPENDENCY_RESTORE.md').write_text(
        '# Round03 source restoration\n\n'
        'ROUND03_ARCHIVE_PREPARATION.json maps every archived executable/config/statistics dependency '
        'to its original absolute path and SHA256. Restore that layout and verify against '
        'round03_transfer/FROZEN_MANIFEST.json. data_metadata contains statistics/hashes only. '
        'The exact runtime/package versions are recorded in ENVIRONMENT.json.\n\n'
        'Provide separately preserved raw datasets, identity metadata and original outer splits '
        'whose paths and hashes are recorded. The internal TRAIN split generator must reproduce '
        'the sealed fit/calibration identities; do not regenerate the outer split. No identity '
        'membership, index arrays, model or optimizer data is in this archive.\n\n'
        'prepare_split_statistics.py regenerates the allowed internal partition and fit-only '
        'normalization under the recorded policy. The fixed33 clean source uses random initialization; '
        'the existing eta0 checkpoint is needed only by the separate frozen affine comparison. '
        'All commands and exact publication/review/admission gates are in round03_transfer/PROTOCOL.md.\n\n'
        'Persistent monitor/register code accepts arbitrary completed_epoch fields and train-only '
        'status/log progress; no validation fields or20epoch assumption are required.\n')
    (ROOT / 'ops/ROUND03_PREPARATION_ALLOWLIST.txt').write_text('\n'.join(sorted(names)) + '\n')
    files = inspect(ROOT, sorted(names))
    print(json.dumps({'files': len(files), 'bytes': sum(x['bytes'] for x in files),
                      'verified_lightweight_dependencies': len(mapping), 'excluded_server_only': excluded,
                      'paths': [x['path'] for x in files]}, indent=2))

if __name__ == '__main__':
    main()
