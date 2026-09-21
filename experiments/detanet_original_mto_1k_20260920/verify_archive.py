"""Read-only verification of published payload hashes and experiment status."""
from pathlib import Path
import hashlib
import json


def verify(root):
    root = root.resolve()
    inventory = json.loads((root / 'archive_manifests/remote_sha256.json').read_text())
    errors = []
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'missing/unsafe: {name}')
            continue
        with path.open('rb') as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            errors.append(f'payload mismatch: {name}')
    for kind in ('evidence', 'checkpoints'):
        manifest = json.loads((root / f'archive_manifests/{kind}_sha256.json').read_text())
        for name, digest in manifest.items():
            if name not in inventory or inventory[name]['sha256'] != digest:
                errors.append(f'{kind} manifest mismatch: {name}')
    summary = json.loads((root / 'reports/instrumented_v2_1299173/diagnostic_summary.json').read_text())
    marker = json.loads((root / 'reports/instrumented_v2_1299173/DIAGNOSTICS_COMPLETE.json').read_text())
    assert summary['main_smoke_passed'] is False
    assert marker['stage_complete'] is False and marker['formal_training_allowed'] is False
    assert summary['main']['detanet_original_uv']['passed'] is False
    assert summary['main']['detanet_mto_planned']['passed'] is True
    assert summary['main']['detanet_mto_global_gate']['passed'] is True
    assert summary['D1_first_attempt_invalidated'] is True
    assert summary['D1_correction']['job_id'] == 1299178
    assert not list(root.rglob('STAGE_COMPLETE'))
    assert not (root / 'reports/PREFLIGHT_PASSED.json').exists()
    result = dict(scope='ARCHIVE_INTEGRITY_ONLY', passed=not errors,
                  files_checked=len(inventory), bytes_checked=sum(v['bytes'] for v in inventory.values()),
                  stage_complete=False, formal_training_submitted=0, errors=errors)
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    verify(Path(__file__).parent)
