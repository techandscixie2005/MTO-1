#!/usr/bin/env python3
"""Freeze an explicit lightweight preparation archive after independent review."""
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, stamp
from package_records import inspect

EXPECTED = '6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e'
REVIEW = '38e0d09ad8d007c5ee7036a9da5fc279ec63019bbca84ee25a0cc8774f4f6452'
PREDECESSOR = '2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae'

def sha(content):
    return hashlib.sha256(content).hexdigest()

def main():
    rd = ROOT / 'round02_frozen'
    mf = json.loads((rd / 'FROZEN_MANIFEST.json').read_text())
    review = json.loads((rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json').read_text())
    assert sha((rd / 'FROZEN_MANIFEST.json').read_bytes()) == EXPECTED
    assert sha((rd / 'INDEPENDENT_PRELAUNCH_REVIEW.json').read_bytes()) == REVIEW
    assert review['passed'] and review['frozen_manifest_sha256'] == EXPECTED
    assert review['source_hashes'] == mf['source_hashes']
    assert review['freshly_verified_source_file_count'] == 64
    assert sha((rd / 'independent_prelaunch_review.py').read_bytes()) == review['review_script_sha256']
    predecessor = json.loads((ROOT / 'ops/ROUND01_PUBLICATION_RECEIPT.json').read_text())
    assert predecessor['remote_verified'] and predecessor['commit'] == PREDECESSOR
    assert not (rd / 'runs').exists(), 'Preparation archive must precede first production launch'
    names, mapping, excluded = set(), [], []
    for original, expected in mf['source_hashes'].items():
        source = Path(original)
        if source.suffix not in {'.py', '.json', '.md'} or source.name == 'splits.json':
            excluded.append({'server_path': original, 'sha256': expected,
                             'reason': 'server-only data, split membership, or checkpoint; independently verified'})
            continue
        content = source.read_bytes()
        assert sha(content) == expected, original
        content.decode('utf-8-sig')
        assert b'\x00' not in content and len(content) < 1024 * 1024, original
        if source.is_relative_to(ROOT):
            relative = source.relative_to(ROOT)
        else:
            parts = source.relative_to('/home/inspur/MTO-1').parts
            relative = Path('round02_dependency_sources', *['data_metadata' if p == 'data' else p for p in parts])
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        name = relative.as_posix()
        names.add(name)
        mapping.append({'server_source': original, 'archived_source': name,
                        'sha256': expected, 'bytes': len(content)})
    names.update('round02_frozen/' + name for name in (
        'FROZEN_MANIFEST.json', 'INDEPENDENT_PRELAUNCH_REVIEW.json', 'independent_prelaunch_review.py',
        'CACHE_EXECUTION.log', 'CACHE_GPU_ADMISSION.xml', 'FREEZE_EXECUTION.log',
        'PREFLIGHT_EXECUTION.log', 'PREFLIGHT_GPU_ADMISSION.xml',
        'RESUME_PREFLIGHT_EXECUTION.log', 'RESUME_PREFLIGHT_INITIAL_DEVICE_CHECK.log',
        'RESUME_GPU_ADMISSION.xml', 'INFERENCE_PREFLIGHT_EXECUTION.log',
        'INFERENCE_PREFLIGHT_INITIAL_MODE_CHECK.log'))
    names.update((
        'FROZEN_MANIFEST.json', 'current_state/COORDINATOR.md', 'current_state/monitor.md',
        'current_state/ROUND01_DECISION.md', 'ops/ROUND01_PUBLICATION_RECEIPT.json',
        'ops/prepare_round02_archive.py', 'ops/package_records.py', 'ops/download_records.py',
        'ops/publish_records.py', 'ops/verify_staged_records.py', 'ops/PUBLISH_COMMAND.ps1',
        'ops/ssh_proxy.py', 'ops/monitor.py', 'ops/register_run.py', 'ops/registry.json',
        'ops/FAILED_ROUND_ARCHIVE_POLICY.md', 'ops/ROUND02_ARCHIVE_PREPARATION.json',
        'ops/ROUND02_DEPENDENCY_RESTORE.md', 'ops/ROUND02_PREPARATION_ALLOWLIST.txt'))
    atomic_json(ROOT / 'ops/ROUND02_ARCHIVE_PREPARATION.json', {
        'prepared_at_utc': stamp(), 'frozen_manifest_sha256': EXPECTED,
        'independent_review_sha256': REVIEW, 'independent_review_passed': True,
        'expected_publication_parent': PREDECESSOR,
        'source_mapping': mapping, 'server_only_hash_references': excluded,
        'lightweight_source_hashes_verified': len(mapping),
        'raw_dataset_checkpoint_cache_or_split_contents_copied': False,
        'production_fits_launched': False, 'download_required_before_staging': True})
    (ROOT / 'ops/ROUND02_DEPENDENCY_RESTORE.md').write_text(
        '# Round02 source restoration\n\n'
        'ROUND02_ARCHIVE_PREPARATION.json maps each exact archived source/config/statistics file '
        'to its original absolute server path and SHA256. Restore this layout and verify hashes '
        'against round02_frozen/FROZEN_MANIFEST.json before reproducing. data_metadata maps back '
        'to data and contains only statistics/hashes. No raw data, split membership or weights are included.\n\n'
        'Provide separately preserved server data, frozen split and starting checkpoint identified by '
        'manifest hashes. Do not regenerate splits. CACHE_COMPLETE.json records reproducible TRAIN cache '
        'construction, checksums and parity; recreate cache with prepare_cache.py when needed. '
        'Cache arrays remain server-only. ENVIRONMENT.json records the pinned interpreter/packages.\n\n'
        'The root PROTOCOL.md and source files are frozen parent dependencies. Round02 objectives, '
        'commands and launch gates are in round02_frozen/PROTOCOL.md and config.json. '
        'The exact independent review and all initial/preflight logs are preserved.\n')
    (ROOT / 'ops/ROUND02_PREPARATION_ALLOWLIST.txt').write_text('\n'.join(sorted(names)) + '\n')
    inspected = inspect(ROOT, sorted(names))
    print(json.dumps({'files': len(inspected), 'bytes': sum(e['bytes'] for e in inspected),
                      'verified_lightweight_dependencies': len(mapping), 'excluded_server_only': excluded,
                      'paths': [e['path'] for e in inspected]}, indent=2))

if __name__ == '__main__':
    main()
