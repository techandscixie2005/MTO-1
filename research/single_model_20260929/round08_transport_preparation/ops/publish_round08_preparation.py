#!/usr/bin/env python3
"""Reviewable main+research fast-forward using a verified D: archive and private index."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

REPO = Path('D:/MTO/publication/MTO-1')
ROOT = Path('D:/MTO/archives/single_model_20260929')
MAIN = '703c2cd7c68f4068160ab9d3dbbab64d4d5af6ba'
PARENT = MAIN
BRANCH = 'codex/single-model-20260929'
README_SOURCE = 'ops/round08_publication/README.md'
APPROVED_README = 'd60f899dc4e9ae7d3fd34f9b2686599eb1d86e3ba2f4b97f883f18d53a1330c8'
SECRET = re.compile(rb'(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
DENIED = re.compile(r'\.(?:pt|pth|ckpt|bin|npy|npz|parquet|h5|hdf5|pkl|pickle|safetensors)$|/(?:data|datasets|cache|caches|__pycache__|weights|checkpoints)/', re.I)
p = argparse.ArgumentParser()
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--publish', action='store_true')
a = p.parse_args()
archive = a.archive.resolve()
assert archive.is_relative_to(ROOT.resolve())
env = os.environ.copy()
env.update(GIT_INDEX_FILE=str(archive / '.main-publication-index'), GIT_AUTHOR_NAME='Codex',
    GIT_COMMITTER_NAME='Codex', GIT_AUTHOR_EMAIL='codex@openai.com',
    GIT_COMMITTER_EMAIL='codex@openai.com', GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
assert 'ssh.github.com' in env.get('GIT_SSH_COMMAND', ''), 'Use the existing scoped SSH transport'

def git(*args, raw=False):
    result = subprocess.run(['git', '-C', str(REPO), '-c', 'credential.interactive=never', *args],
        env=env, capture_output=True, check=True, timeout=240)
    return result.stdout if raw else result.stdout.decode().strip()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(name, value):
    path = archive / name
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def tips():
    lines = git('ls-remote', '--heads', 'origin', 'refs/heads/main', 'refs/heads/' + BRANCH).splitlines()
    actual = {line.split()[1]: line.split()[0] for line in lines}
    assert actual == {'refs/heads/main': MAIN, 'refs/heads/' + BRANCH: PARENT}, actual
    return actual

def worktree_state():
    return dict(head_sha256=sha((REPO/'.git/HEAD').read_bytes()),
        default_index_exists=(REPO/'.git/index').exists(),
        metadata_sha256=sha((REPO/'PUBLICATION_TREE_IDS.json').read_bytes()))

receipt_bytes = (archive / 'LOCAL_DOWNLOAD_RECEIPT.json').read_bytes()
receipt = json.loads(receipt_bytes)
manifest = json.loads((archive / 'ARCHIVE_MANIFEST.json').read_bytes())
assert receipt['all_member_hashes_verified'] and receipt['download_precedes_git_staging']
assert manifest['round_id'] == 'round08_transport_preparation'
assert sha((archive / README_SOURCE).read_bytes()) == APPROVED_README
records = list(manifest['files'])
for name in ('ARCHIVE_MANIFEST.json', 'LOCAL_DOWNLOAD_RECEIPT.json'):
    data = (archive / name).read_bytes()
    records.append(dict(path=name, bytes=len(data), sha256=sha(data)))
assert any(e['path'] == README_SOURCE for e in records)
for e in records:
    path = PurePosixPath(e['path'])
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in e['path']
    data = (archive / e['path']).read_bytes()
    assert len(data) == e['bytes'] and sha(data) == e['sha256'], e['path']
    assert b'\x00' not in data and not SECRET.search(data), e['path']
    assert not DENIED.search('/' + e['path']), e['path']
    assert not any(part.startswith(('private_partition', 'private_preflight')) for part in path.parts), e['path']
    data.decode('utf-8-sig')

if not a.publish:
    assert not (archive / 'STAGED_REVIEW_MAIN.json').exists(), 'Preserve prior review; diagnose instead of restaging'
    tips()
    untouched = worktree_state()
    git('fetch', '--filter=tree:0', 'origin', 'main', BRANCH)
    git('merge-base', '--is-ancestor', MAIN, PARENT)
    git('read-tree', PARENT)
    staged = []
    for e in records:
        dest = ('README.md' if e['path'] == README_SOURCE else
            'research/single_model_20260929/' + manifest['round_id'] + '/' + e['path'])
        blob = git('hash-object', '-w', '--no-filters', '--', str(archive / e['path']))
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + dest)
        assert git('cat-file', 'blob', blob, raw=True) == (archive / e['path']).read_bytes()
        staged.append(dict(e, git_path=dest, blob=blob))
    changes = git('diff', '--cached', '--name-status', PARENT).splitlines()
    expected = {('M' if f['git_path'] == 'README.md' else 'A') + '\t' + f['git_path'] for f in staged}
    assert set(changes) == expected
    tree = git('write-tree', '--missing-ok')
    diff = git('diff', '--cached', '--no-ext-diff', PARENT, raw=True)
    (archive / 'STAGED_DIFF_MAIN.patch').write_bytes(diff)
    review = dict(prepared_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), parent=PARENT,
        expected_main=MAIN, tree=tree, files=staged, staged_count=len(staged),
        new_records_bytes=sum(f['bytes'] for f in staged), download_receipt_sha256=sha(receipt_bytes),
        diff_sha256=sha(diff), readme_sha256=sha((archive / README_SOURCE).read_bytes()),
        byte_verified=True, all_changes_allowlisted=True, draft_research_excluded=True,
        worktree_state=untouched,
        root_and_independent_content_review_required=True)
    save('STAGED_REVIEW_MAIN.json', review)
    print(json.dumps({k:v for k,v in review.items() if k != 'files'}, indent=2))
else:
    review = json.loads((archive / 'STAGED_REVIEW_MAIN.json').read_bytes())
    approval = json.loads((archive / 'MAIN_CONTENT_APPROVAL.json').read_bytes())
    assert approval['passed'] and approval['tree'] == review['tree'] and approval['parent'] == PARENT
    assert approval['readme_sha256'] == review['readme_sha256']
    assert review['download_receipt_sha256'] == sha(receipt_bytes)
    assert worktree_state() == review['worktree_state']
    assert git('write-tree', '--missing-ok') == review['tree']
    for f in review['files']:
        data = (archive / f['path']).read_bytes()
        assert sha(data) == f['sha256'] and git('cat-file', 'blob', f['blob'], raw=True) == data
    tips()
    message = archive / 'MAIN_COMMIT_MESSAGE.txt'
    message.write_text('Publish reviewed Round08 backbone tensor transport preparation\n\n'
        'Archive source closure, technical checks, failure evidence and reviews after verified D download.\n'
        'Record preparation readiness only; no production or accuracy claim.\n', encoding='utf-8')
    commit = git('commit-tree', review['tree'], '-p', PARENT, '-F', str(message))
    save('MAIN_LOCAL_COMMIT_RECEIPT.json', dict(commit=commit, parent=PARENT, tree=review['tree'], pushed=False))
    tips()
    git('push', '--atomic', 'origin', commit + ':refs/heads/main', commit + ':refs/heads/' + BRANCH)
    lines = git('ls-remote', '--heads', 'origin', 'refs/heads/main', 'refs/heads/' + BRANCH).splitlines()
    remote = {line.split()[1]:line.split()[0] for line in lines}
    assert remote == {'refs/heads/main':commit, 'refs/heads/' + BRANCH:commit}, remote
    assert worktree_state() == review['worktree_state']
    git('fetch', '--filter=tree:0', 'origin', 'main', BRANCH)
    assert git('rev-parse', 'refs/remotes/origin/main') == commit
    assert git('rev-parse', 'refs/remotes/origin/' + BRANCH) == commit
    for f in review['files']:
        assert git('cat-file', 'blob', 'refs/remotes/origin/main:' + f['git_path'], raw=True) == (archive / f['path']).read_bytes()
    publication = dict(published_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), commit=commit,
        parent=PARENT, previous_main=MAIN, tree=review['tree'], branches=['main',BRANCH],
        remote_verified=True, all_new_remote_blob_bytes_verified=True, atomic_non_force_push=True, download_before_stage_before_commit_push=True,
        source_manifest_sha256='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146',
        frozen_manifest_sha256='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146',
        independent_review_sha256='87de2d873f2a24033a2f646ac825f26abbbae7eadbd5305a723fbb6a17093e8b',
        root_acceptance_sha256=sha((archive/'current_state/ROUND08_PREPARATION_ACCEPTANCE.md').read_bytes()),
        preparation_only=True, production_execution_authorized=False,
        report_sha256=sha((archive/'round08_transport_preparation/PREPARATION_REPORT.md').read_bytes()),
        archive_directory=str(archive), bundle_sha256=receipt['bundle_sha256'],
        archive_file_count=receipt['file_count'], archive_total_bytes=manifest['total_bytes'],
        readme_sha256=review['readme_sha256'], staged_review_sha256=sha((archive/'STAGED_REVIEW_MAIN.json').read_bytes()),
        content_approval_sha256=sha((archive/'MAIN_CONTENT_APPROVAL.json').read_bytes()),
        working_tree_and_head_untouched=True)
    save('MAIN_PUBLICATION_RECEIPT.json', publication)
    print(json.dumps(publication, indent=2))
