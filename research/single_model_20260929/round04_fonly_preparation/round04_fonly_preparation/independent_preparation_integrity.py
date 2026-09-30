"""Reviewer-only text/hash closure audit; no model or array imports/reads."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MANIFEST = '79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def constants(path):
    tree = ast.parse(path.read_text())
    return {n.targets[0].id: ast.dump(n.value, include_attributes=False)
            for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)}


def main():
    manifest_path = ROOT/'FROZEN_MANIFEST.json'
    assert sha(manifest_path) == EXPECTED_MANIFEST
    manifest = read(manifest_path)
    assert manifest['phase'] == 'preparation_only'
    assert manifest['production_execution_authorized'] is False
    for path, expected in manifest['source_hashes'].items():
        assert Path(path).suffix in ('.py', '.json', '.md')
        assert sha(path) == expected, path
    for folder in ('documents_only_snapshot', 'stub_preparation_snapshot'):
        for name, expected in manifest[folder].items():
            assert sha(ROOT/folder/name) == expected, (folder, name)
    for name, expected in manifest['preflight_receipt_hashes'].items():
        assert sha(ROOT/name) == expected
        assert read(ROOT/name)['passed'] is True
    # These two final-code receipts bind all current paths they exercised.
    for name in ('INDEPENDENT_GATE_CHECKS.json', 'PREFLIGHT_CONTINUITY.json'):
        for path, expected in read(ROOT/name)['source_hashes'].items():
            assert sha(ROOT/path) == expected, (name, path)
    assert constants(ROOT/'scalar_map.py') == constants(ROOT/'stub_preparation_snapshot/scalar_map.py')
    assert not (ROOT/'production').exists()
    assert not (ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    result = {
        'passed': True,
        'scope': 'preparation text/hash closure only',
        'source_manifest_sha256': EXPECTED_MANIFEST,
        'source_entry_count': len(manifest['source_hashes']),
        'source_hashes': manifest['source_hashes'],
        'preflight_receipt_hashes': manifest['preflight_receipt_hashes'],
        'prior_snapshots_hashes_verified': True,
        'current_adversarial_and_continuity_receipt_sources_match': True,
        'mathematical_module_constants_unchanged': True,
        'binary_references_rehashed_by_this_audit': False,
        'binary_reference_note': 'Inherited audited input hashes; future authorized gate rehashes them. No binary input was opened by this script.',
        'production_directory_exists': False,
        'production_execution_authorization_exists': False,
        'new_real_target_fit_or_validation_access': False,
        'review_script_sha256': sha(__file__),
    }
    text = json.dumps(result, indent=2) + '\n'
    (ROOT/'INDEPENDENT_PREPARATION_INTEGRITY.json').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
