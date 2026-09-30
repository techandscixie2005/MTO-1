#!/usr/bin/env python3
"""Package preparation records only after exact review and root archival closeout."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops'))
from monitor import ROOT, atomic_json, safe_path, stamp
from package_records import inspect

RD = ROOT / 'round04_fonly_preparation'
EXPECTED_SOURCE = '79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f'
EXPECTED_REVIEW = 'b7953db3597dc98eebe820a3c44341b933cc16128e02c22672a114cdd60e5d49'
EXPECTED_REVIEW_MANIFEST = '018fbb2b54eca2b2d2eaaf8815c39171af9190b55c587fb66a81d02e8ffaa13b'
EXPECTED_CLOSEOUT = '191d28eab7e5af3e6bd417b0f3ea354588a9d9f0a011e9dba8c24baf7d9c0a84'
PARENT = '4f9ae50647f957ac7d68ab1a20200a79fa527cf9'
EXTRA_TOP_LEVEL = (
    'PROPOSAL_MANIFEST.json', 'PREPARATION_RESULTS.md', 'FROZEN_MANIFEST.json',
    'INDEPENDENT_REVIEW_MANIFEST.json', 'INDEPENDENT_PREPARATION_REVIEW.json',
    'INDEPENDENT_PREPARATION_REVIEW.md', 'independent_preparation_integrity.py',
    'INDEPENDENT_PREPARATION_INTEGRITY.json', 'INDEPENDENT_PREPARATION_INTEGRITY_01.log',
    'DESIGN_AUDIT_01.log', 'INFERENCE_PREFLIGHT_01.log', 'SYNTHETIC_CHECKS_01.log',
    'SYNTHETIC_CHECKS_02.log', 'SYNTHETIC_CHECKS_03.log', 'PRODUCTION_GUARD_CHECKS_01.log',
    'PREFLIGHT_CONTINUITY_01.log', 'INDEPENDENT_GATE_CHECKS_01.log', 'FREEZE_PREPARATION_01.log',
)
OPERATIONAL = (
    'current_state/ROUND04_PREPARATION_DECISION.md', 'current_state/ROUND04_PREPARATION_CLOSEOUT.md',
    'current_state/COORDINATOR.md', 'current_state/monitor.md', 'current_state/science_implementation.md',
    'monitoring/CHECK_20260929T234954Z.json', 'monitoring/CLOSED_CAMPAIGN_20260929T234953.json',
    'monitoring/ROUND03_PUBLICATION_WORDING_CORRECTION_20260929T2350.json',
    'ops/HEARTBEAT_CLOSED_CHECK_20260929T2350.json', 'ops/record_completed_campaign_check.py',
    'ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json', 'ops/package_records.py', 'ops/download_records.py',
    'ops/publish_records.py', 'ops/verify_staged_records.py', 'ops/PUBLISH_COMMAND.ps1',
    'ops/ssh_proxy.py', 'ops/FAILED_ROUND_ARCHIVE_POLICY.md',
    'round04_fonly_preparation/ops/ARCHIVE_CHECKLIST.md',
    'round04_fonly_preparation/ops/PROSPECTIVE_RECORDS.json',
    'round04_fonly_preparation/ops/INHERITED_DEPENDENCY_VERIFICATION.json',
    'round04_fonly_preparation/ops/HEARTBEAT_BEFORE_PUBLICATION.toml',
    'round04_fonly_preparation/ops/finalize_preparation_archive.py',
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def main():
    assert sha(RD / 'FROZEN_MANIFEST.json') == EXPECTED_SOURCE
    assert sha(RD / 'INDEPENDENT_PREPARATION_REVIEW.json') == EXPECTED_REVIEW
    assert sha(RD / 'INDEPENDENT_REVIEW_MANIFEST.json') == EXPECTED_REVIEW_MANIFEST
    assert sha(ROOT / 'current_state/ROUND04_PREPARATION_CLOSEOUT.md') == EXPECTED_CLOSEOUT
    manifest = read(RD / 'FROZEN_MANIFEST.json')
    review = read(RD / 'INDEPENDENT_PREPARATION_REVIEW.json')
    review_manifest = read(RD / 'INDEPENDENT_REVIEW_MANIFEST.json')
    integrity = read(RD / 'INDEPENDENT_PREPARATION_INTEGRITY.json')
    assert manifest['phase'] == review_manifest['phase'] == 'preparation_only'
    assert review['passed'] and integrity['passed'] and not review['blocking_findings']
    assert manifest['source_hashes'] == review['source_hashes'] == integrity['source_hashes']
    assert len(manifest['source_hashes']) == 51
    assert review['source_manifest_sha256'] == review_manifest['source_manifest_sha256'] == EXPECTED_SOURCE
    for record in (manifest, review, review_manifest):
        assert record['production_execution_authorized'] is False
    assert not review['fit_authorized'] and not review['outer_validation_value_access_authorized']
    assert not review['test_access_authorized'] and not review['execution_authorization_present']
    assert review['subsequent_scope_authorizing_instruction_required']
    assert manifest['actual_real_data_solves'] == manifest['actual_validation_comparisons'] == 0
    assert not manifest['production_artifacts_present'] and not (RD / 'production').exists()
    assert not (RD / 'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    forbidden = [p.relative_to(RD).as_posix() for p in RD.rglob('*') if p.is_file() and p.suffix.lower() in
        {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.pkl', '.pickle', '.safetensors'}]
    assert not forbidden, forbidden
    inherited = read(RD / 'ops/INHERITED_DEPENDENCY_VERIFICATION.json')
    assert inherited['parent_commit'] == PARENT and not inherited['raw_coefficients_reexported']
    inherited_by_path = {item['server_path']: item for item in inherited['files']}
    names = set(OPERATIONAL)
    mapping = []
    for original, expected in manifest['source_hashes'].items():
        source = Path(original)
        assert source.suffix in {'.py', '.json', '.md'}
        assert sha(source) == expected, original
        if original in inherited_by_path:
            entry = inherited_by_path[original]
            assert entry['sha256'] == expected
            mapping.append({'server_path': original, 'sha256': expected,
                'storage': 'inherited_published_parent', 'parent_commit': PARENT,
                'git_path': entry['git_path'], 'copied_to_new_archive': False})
            continue
        if source.is_relative_to(ROOT):
            relative = source.relative_to(ROOT)
        else:
            components = source.relative_to('/home/inspur/MTO-1').parts
            relative = Path('round04_fonly_preparation/dependency_sources',
                *['data_metadata' if part == 'data' else part for part in components])
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes()
            content.decode('utf-8-sig')
            assert len(content) < 1024 * 1024 and b'\x00' not in content
            if target.exists():
                assert target.read_bytes() == content, str(target)
            else:
                target.write_bytes(content)
        names.add(relative.as_posix())
        mapping.append({'server_path': original, 'sha256': expected,
            'storage': 'this_archive', 'archive_path': relative.as_posix()})
    for group in ('documents_only_snapshot', 'stub_preparation_snapshot'):
        for member, expected in manifest[group].items():
            path = RD / group / member
            assert sha(path) == expected, str(path)
            names.add(path.relative_to(ROOT).as_posix())
    for member, expected in manifest['preflight_receipt_hashes'].items():
        assert sha(RD / member) == expected, member
        names.add((RD / member).relative_to(ROOT).as_posix())
    for original, expected in review_manifest['files'].items():
        path = Path(original)
        assert sha(path) == expected, original
        names.add(path.relative_to(ROOT).as_posix())
    for member in EXTRA_TOP_LEVEL:
        names.add((RD / member).relative_to(ROOT).as_posix())
    assert sha(RD / 'INDEPENDENT_PREPARATION_INTEGRITY.json') == review['integrity_receipt_sha256']
    assert sha(RD / 'INDEPENDENT_PREPARATION_REVIEW.md') == review['review_markdown_sha256']
    restore = RD / 'ops/DEPENDENCY_RESTORE.md'
    restore.write_text('# Round04 preparation restoration\n\n'
        'ARCHIVE_INVENTORY.json maps all51 exact source/dependency records to their original absolute paths. '
        'Forty-nine are copied as lightweight source/config/metadata records. Two historical coefficient/result '
        'dependencies are inherited from the exact published Round03 completion paths at commit ' + PARENT + '. '
        'INHERITED_DEPENDENCY_VERIFICATION.json verifies their Git object bytes against the source manifest. '
        'Restore these inherited text dependencies from that parent and verify their SHA256; they are not recopied here.\n\n'
        'Restore archived source/config/statistics to the mapped original paths and use ENVIRONMENT.json '
        'and RUNNER_HANDOFF.md for the pinned environment and commands. data_metadata contains normalization '
        'statistics, not datasets. The separate binary_input_hash_references in the frozen manifest name '
        'server-only checkpoints, datasets and prediction caches; no such binary is in this archive. '
        'Do not decode validation values or run production while restoring preparation records.\n\n'
        'The preserved documents_only_snapshot and stub_preparation_snapshot are historical phases. '
        'The final source manifest and independent review describe current readiness. Publication and '
        'preflight PASS grant no fit/evaluation authority: a later user instruction and a distinct bound '
        'execution decision are required. No execution authorization is generated by this archive.\n')
    names.add(restore.relative_to(ROOT).as_posix())
    files = inspect(ROOT, sorted(names))
    inventory = {'prepared_at_utc': stamp(), 'phase': 'preparation_only_reviewed_ready_to_archive',
        'source_manifest_sha256': EXPECTED_SOURCE, 'frozen_manifest_sha256': EXPECTED_SOURCE,
        'independent_review_sha256': EXPECTED_REVIEW, 'independent_review_manifest_sha256': EXPECTED_REVIEW_MANIFEST,
        'root_closeout_sha256': EXPECTED_CLOSEOUT, 'parent_commit': PARENT, 'source_dependency_mapping': mapping,
        'server_only_binary_hash_references': manifest['binary_input_hash_references'],
        'source_records_mapped': len(mapping), 'inherited_records_not_recopied': len(inherited_by_path),
        'snapshot_groups_verified': ['documents_only_snapshot', 'stub_preparation_snapshot'],
        'files_before_inventory_and_allowlist': files, 'production_execution_authorized': False,
        'actual_real_data_solves': 0, 'actual_validation_comparisons': 0,
        'checkpoints_tensors_learned_arrays_raw_data_caches_credentials_exported': False,
        'user_instruction_required_before_future_production': True,
        'required_order': ['server explicit lightweight package', 'FIRST D download and every hash verification',
            'exact-byte stage/content inspection', 'independent staged review', 'non-force commit/push',
            'remote verification', 'current handoffs and existing heartbeat update only after publication']}
    atomic_json(RD / 'ops/ARCHIVE_INVENTORY.json', inventory)
    names.update(('round04_fonly_preparation/ops/ARCHIVE_INVENTORY.json',
        'round04_fonly_preparation/ops/ARCHIVE_ALLOWLIST.txt'))
    (RD / 'ops/ARCHIVE_ALLOWLIST.txt').write_text('\n'.join(sorted(names)) + '\n')
    files = inspect(ROOT, sorted(names))
    print(json.dumps({'phase': inventory['phase'], 'files': len(files),
        'bytes': sum(item['bytes'] for item in files), 'source_records_mapped': len(mapping),
        'inherited_records_not_recopied': len(inherited_by_path), 'production_execution_authorized': False}))

if __name__ == '__main__':
    main()
