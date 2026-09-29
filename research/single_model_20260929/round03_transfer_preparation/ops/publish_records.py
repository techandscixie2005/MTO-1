#!/usr/bin/env python3
"""Stage verified downloaded text records in a separate Git index; optional push."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

REPO = Path('D:/MTO/publication/MTO-1')
BRANCH = 'codex/single-model-20260929'
BASE = '69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e'
PREFIX = 'research/single_model_20260929'
p = argparse.ArgumentParser()
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--publish', action='store_true', help='Only after human/agent review of STAGED_REVIEW and diff')
p.add_argument('--remote-url', default='origin')
p.add_argument('--http-proxy', default=None)
a = p.parse_args()
archive = a.archive.resolve()
if not archive.is_relative_to(Path('D:/MTO/archives/single_model_20260929').resolve()):
    raise SystemExit('Must publish from the downloaded D: archive')
receipt = json.loads((archive / 'LOCAL_DOWNLOAD_RECEIPT.json').read_text())
manifest = json.loads((archive / 'ARCHIVE_MANIFEST.json').read_text())
if not receipt['all_member_hashes_verified'] or not receipt['download_precedes_git_staging']:
    raise SystemExit('Missing verified download-first evidence')
index = archive / '.publication-index'
env = os.environ.copy()
env['GIT_INDEX_FILE'] = str(index)
env['GIT_AUTHOR_NAME'] = env['GIT_COMMITTER_NAME'] = 'Codex'
env['GIT_AUTHOR_EMAIL'] = env['GIT_COMMITTER_EMAIL'] = 'codex@openai.com'
env['GIT_TERMINAL_PROMPT'] = '0'
env['GCM_INTERACTIVE'] = 'never'


def git(*args, raw=False, input=None):
    prefix = ['git', '-C', str(REPO), '-c', 'credential.interactive=never']
    if a.http_proxy:
        prefix += ['-c', 'http.proxy=' + a.http_proxy]
    if a.remote_url.startswith('https://github.com/'):
        # Partial-clone lazy fetches use origin, so apply the same transport
        # to child Git processes without changing repository/global config.
        prefix += ['-c', 'url.https://github.com/.insteadOf=git@github.com:']
    r = subprocess.run([*prefix, *args], env=env, input=input, capture_output=True, check=True)
    return r.stdout if raw else r.stdout.decode().strip()


files = list(manifest['files'])
for name in ('ARCHIVE_MANIFEST.json', 'LOCAL_DOWNLOAD_RECEIPT.json'):
    data = (archive / name).read_bytes()
    files.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
for e in files:
    q = PurePosixPath(e['path'])
    if q.is_absolute() or '..' in q.parts or '\\' in e['path']:
        raise SystemExit('Unsafe record path')
    data = archive.joinpath(*q.parts).read_bytes()
    if len(data) != e['bytes'] or hashlib.sha256(data).hexdigest() != e['sha256']:
        raise SystemExit('Archive record changed: ' + e['path'])
    data.decode('utf-8-sig')
    if b'\x00' in data:
        raise SystemExit('Binary record rejected')

review_path = archive / 'STAGED_REVIEW.json'
if not a.publish:
    git('fetch', '--filter=tree:0', a.remote_url, 'main')
    if git('rev-parse', 'FETCH_HEAD') != BASE:
        raise SystemExit('origin/main differs from audited base; coordinator must review')
    remote = git('ls-remote', '--heads', a.remote_url, 'refs/heads/' + BRANCH)
    parent = remote.split()[0] if remote else BASE
    if remote:
        git('fetch', '--filter=tree:0', a.remote_url, BRANCH)
    git('read-tree', parent)
    staged = []
    for e in files:
        src = archive.joinpath(*PurePosixPath(e['path']).parts)
        blob = git('hash-object', '-w', '--no-filters', '--', str(src))
        dest = PREFIX + '/' + manifest['round_id'] + '/' + e['path']
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + dest)
        staged.append(dict(e, git_path=dest, blob=blob))
    names = git('diff', '--cached', '--name-only', parent).splitlines()
    expected = {e['git_path'] for e in staged}
    if not set(names) <= expected:
        raise SystemExit('Unexpected staged names')
    forbidden = re.compile(r'\.(?:pt|pth|ckpt|bin|npy|npz|parquet|h5|hdf5|pkl|pickle|safetensors)$|/(?:data|datasets|cache|caches|__pycache__|weights|checkpoints)/', re.I)
    if any(forbidden.search(name) for name in names):
        raise SystemExit('Forbidden staged path')
    diff = git('diff', '--cached', '--no-ext-diff', '--stat', parent)
    (archive / 'STAGED_DIFF.patch').write_bytes(git('diff', '--cached', '--no-ext-diff', parent, raw=True))
    tree = git('write-tree', '--missing-ok')
    review = {'prepared_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'archive': str(archive),
              'download_receipt_sha256': hashlib.sha256((archive / 'LOCAL_DOWNLOAD_RECEIPT.json').read_bytes()).hexdigest(),
              'branch': BRANCH, 'parent': parent, 'tree': tree, 'staged_count': len(names),
              'new_records_bytes': sum(e['bytes'] for e in staged), 'files': staged, 'diff_stat': diff,
              'human_or_agent_content_review_required_before_publish': True}
    review_path.write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'review': str(review_path), 'diff': str(archive / 'STAGED_DIFF.patch'), 'staged_files': len(names), 'parent': parent, 'tree': tree, 'diff_stat': diff}, indent=2))
else:
    review = json.loads(review_path.read_text())
    content_review = json.loads((archive / 'CONTENT_REVIEW.json').read_text())
    if not content_review['passed'] or content_review['tree'] != review['tree'] or content_review['parent'] != review['parent']:
        raise SystemExit('Staged content review does not match the current reviewed tree')
    if git('write-tree', '--missing-ok') != review['tree']:
        raise SystemExit('Index changed after preparation')
    if hashlib.sha256((archive / 'LOCAL_DOWNLOAD_RECEIPT.json').read_bytes()).hexdigest() != review['download_receipt_sha256']:
        raise SystemExit('Download receipt changed')
    remote = git('ls-remote', '--heads', a.remote_url, 'refs/heads/' + BRANCH)
    remote_parent = remote.split()[0] if remote else BASE
    if remote_parent != review['parent']:
        raise SystemExit('Remote advanced; prepare and review again')
    message = ('Archive MTO single-model ' + manifest['round_id'] + '\n\n'
               'Publish reviewed lightweight research records after verified server-to-D: download.\n'
               'Preserve one-model evaluation and scoped four-hour monitoring provenance.\n')
    message_path = archive / 'COMMIT_MESSAGE.txt'
    message_path.write_text(message)
    commit = git('commit-tree', review['tree'], '-p', review['parent'], '-F', str(message_path))
    (archive / 'LOCAL_COMMIT_RECEIPT.json').write_text(json.dumps({'commit': commit,
        'parent': review['parent'], 'tree': review['tree'], 'push_verified': False}, indent=2) + '\n')
    print(json.dumps({'local_commit_created': commit}), flush=True)
    git('push', a.remote_url, commit + ':refs/heads/' + BRANCH)
    verified = git('ls-remote', '--heads', a.remote_url, 'refs/heads/' + BRANCH).split()[0]
    if verified != commit:
        raise SystemExit('Remote verification failed')
    publication = {'published_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'branch': BRANCH,
                   'commit': commit, 'parent': review['parent'], 'tree': review['tree'],
                   'download_before_stage_before_commit_push': True, 'remote_verified': True,
                   'staged_review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
                   'working_tree_and_head_untouched': True}
    (archive / 'PUBLICATION_RECEIPT.json').write_text(json.dumps(publication, indent=2) + '\n')
    print(json.dumps(publication, indent=2))
