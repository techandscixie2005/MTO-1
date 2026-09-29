#!/usr/bin/env python3
"""Package an explicit reviewed allowlist of lightweight research records."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from monitor import ROOT, safe_path, stamp

EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.txt', '.csv', '.log', '.sh', '.ps1', '.yaml', '.yml', '.toml', '.xml', '.diff', '.patch'}
DENIED_PARTS = {'data', 'datasets', 'cache', 'caches', '__pycache__', '.git', '.ssh', 'weights', 'checkpoints'}
SECRET = re.compile(r'(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def inspect(root, names):
    result, total = [], 0
    for name in sorted(set(names)):
        p = safe_path(root, name)
        if Path(name).is_absolute() or str(Path(name)) != name:
            raise ValueError('Only canonical relative names accepted: ' + name)
        if p.is_symlink() or not p.is_file():
            raise ValueError('Missing/nonregular file: ' + name)
        if p.suffix.lower() not in EXTENSIONS or set(Path(name).parts) & DENIED_PARTS:
            raise ValueError('Forbidden path/extension: ' + name)
        if p.name != 'dataset.py' and re.search(r'(?:prediction|dataset|credential|optimizer_state|model_weight)', p.name, re.I):
            raise ValueError('Forbidden artifact name: ' + name)
        content = p.read_bytes()
        if len(content) > 5 * 1024 * 1024:
            raise ValueError('File exceeds 5 MiB: ' + name)
        text = content.decode('utf-8-sig')
        if '\x00' in text or SECRET.search(text):
            raise ValueError('Binary/credential content rejected: ' + name)
        total += len(content)
        result.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    if total > 25 * 1024 * 1024:
        raise ValueError('Bundle exceeds 25 MiB')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--allowlist', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round-id', required=True)
    a = p.parse_args()
    names = [line.strip() for line in a.allowlist.read_text().splitlines() if line.strip() and not line.startswith('#')]
    entries = inspect(a.root, names)
    manifest = {'round_id': a.round_id, 'packaged_at_utc': stamp(), 'source_root': str(a.root),
                'policy': 'Explicit text-only allowlist; no checkpoints, weights, optimizer states, raw data, caches, credentials.',
                'files': entries, 'total_bytes': sum(e['bytes'] for e in entries)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, 'w:gz') as tar:
        for e in entries:
            content = (a.root / e['path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != e['sha256']:
                raise ValueError('File changed while packaging: ' + e['path'])
            info = tarfile.TarInfo(e['path'])
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    print(json.dumps({'bundle': str(a.output), 'sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
                      'files': len(entries), 'total_bytes': manifest['total_bytes']}))


if __name__ == '__main__':
    main()
