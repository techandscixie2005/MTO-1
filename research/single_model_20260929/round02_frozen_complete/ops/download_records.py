#!/usr/bin/env python3
"""Download server bundle FIRST, verify, and extract within D:\\MTO\\archives."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--remote-bundle', required=True)
p.add_argument('--expected-sha256', required=True)
p.add_argument('--round-id', required=True)
a = p.parse_args()
if not a.round_id.replace('_', '').replace('-', '').isalnum():
    raise SystemExit('Invalid round id')
archive = Path('D:/MTO/archives/single_model_20260929') / a.round_id
if archive.exists():
    raise SystemExit('Archive exists: choose distinct id; never overwrite prior receipt')
archive.mkdir(parents=True)
bundle = archive / 'server_records.tar.gz'
subprocess.run(['scp', 'USTC-A800:' + a.remote_bundle, str(bundle)], check=True)
if hashlib.sha256(bundle.read_bytes()).hexdigest() != a.expected_sha256:
    raise SystemExit('Downloaded bundle hash mismatch; preserve for diagnosis')
with tarfile.open(bundle, 'r:gz') as tar:
    members = tar.getmembers()
    names = [m.name for m in members]
    if len(names) != len(set(names)) or 'ARCHIVE_MANIFEST.json' not in names:
        raise SystemExit('Invalid manifest or duplicate members')
    for m in members:
        q = PurePosixPath(m.name)
        if not m.isfile() or q.is_absolute() or '..' in q.parts or '\\' in m.name:
            raise SystemExit('Unsafe archive member')
    manifest = json.load(tar.extractfile('ARCHIVE_MANIFEST.json'))
    allowed = {entry['path']: entry for entry in manifest['files']}
    if set(names) != set(allowed) | {'ARCHIVE_MANIFEST.json'}:
        raise SystemExit('Bundle members differ from manifest')
    for m in members:
        data = tar.extractfile(m).read()
        if m.name in allowed:
            e = allowed[m.name]
            if len(data) != e['bytes'] or hashlib.sha256(data).hexdigest() != e['sha256']:
                raise SystemExit('Member hash/size mismatch: ' + m.name)
        target = archive.joinpath(*PurePosixPath(m.name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
receipt = {'downloaded_verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
           'remote_bundle': a.remote_bundle, 'bundle_sha256': a.expected_sha256,
           'archive_directory': str(archive), 'file_count': len(allowed),
           'all_member_hashes_verified': True, 'download_precedes_git_staging': True,
           'publication_state': 'not_staged_or_committed_by_this_script'}
(archive / 'LOCAL_DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
