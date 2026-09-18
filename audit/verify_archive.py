"""Verify the archived source and server artifacts without importing PyTorch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    expected = json.loads((ROOT / 'configs/frozen.json').read_text(encoding='utf-8-sig'))['files']
    for line in (ROOT / 'docs/server-artifact-sha256.txt').read_text(encoding='utf-8-sig').splitlines():
        sha, name = line.split('  ', 1)
        expected[name] = sha
    failures = []
    for name, sha in expected.items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f'MISSING {name}')
        elif digest(path) != sha:
            failures.append(f'MISMATCH {name}')
    print(json.dumps({'checked': len(expected), 'passed': not failures, 'failures': failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
