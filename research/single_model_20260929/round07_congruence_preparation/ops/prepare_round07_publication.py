#!/usr/bin/env python3
"""Exact lightweight Round07 archival closure; no scientific imports or execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD = ROOT / 'round07_congruence_preparation'
OP = ROOT / 'ops/round07_publication'
SOURCE = '68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203'
REVIEW = 'e3ae0804141dc37fdbcb3797411a301e75dabe223eea4005d0c043197f423826'
REVIEW_MANIFEST = '459d1a35dc2f39ab02f64b600c0731c391cd633584112e25b94bfe53d49b4687'
ACCEPTANCE = '89ffc0d7d04b99eecafdebdf9cf54fea2a89b0522ec8fd1e2bf1a20dfe704826'
README = 'd926ffbc554e2376ab311b875eecedb9c00f717f9a3752a5ad384b4c48fde35d'


def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path, expected=None):
    assert path.is_file() and not path.is_symlink() and path.resolve() == path, str(path)
    data = path.read_bytes()
    if expected:
        assert sha(data) == expected, str(path)
    return data

def save(path, data):
    assert not path.exists(), 'Preserve previous operational artifact: ' + str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def encoded(obj):
    return (json.dumps(obj, indent=2) + '\n').encode()

def inspect(path):
    data = read(path)
    assert path.suffix.lower() in EXTENSIONS, str(path)
    assert not set(path.parts) & DENIED_PARTS, str(path)
    assert not any(p.startswith(('private_partition', 'private_preflight')) for p in path.parts), str(path)
    if path.name != 'dataset.py' and re.search(r'(?:prediction|dataset|credential|optimizer_state|model_weight)', path.name, re.I):
        raise AssertionError('Forbidden artifact name: ' + str(path))
    assert len(data) <= 5 * 1024 * 1024
    text = data.decode('utf-8-sig')
    assert '\x00' not in text and not SECRET.search(text), str(path)
    assert path.suffix.lower() != '.svg', 'No SVG needed in this preparation archive'
    return data

def bind():
    source = json.loads(read(RD/'FROZEN_MANIFEST.json', SOURCE))
    review = json.loads(read(RD/'INDEPENDENT_PREPARATION_REVIEW.json', REVIEW))
    rm = json.loads(read(RD/'INDEPENDENT_REVIEW_MANIFEST.json', REVIEW_MANIFEST))
    read(ROOT/'current_state/ROUND07_PREPARATION_ACCEPTANCE.md', ACCEPTANCE)
    read(OP/'README.md', README)
    assert review['passed'] and review['frozen_manifest_sha256'] == SOURCE
    assert not review['production_authorized'] and len(source['source_hashes']) == 185
    assert source['actual_discarded_optimizer_updates']==9 and source['preparation_only_no_production_authorization']
    assert review['source_hashes'] == source['source_hashes']
    assert rm['source_manifest_sha256'] == SOURCE and rm['independent_review_sha256'] == REVIEW
    assert rm['phase']=='preparation_only' and not rm['production_execution_authorized']
    assert len(rm['files'])==9
    assert review['report_sha256']==sha(read(RD/'PREPARATION_REPORT.md'))
    return source, rm

def prepare():
    source, rm = bind()
    assert not (OP/'ARCHIVE_INVENTORY.json').exists()
    wanted = {}
    def add(path, expected=None):
        data = inspect(path)
        if expected:
            assert sha(data) == expected, str(path)
        if path in wanted:
            assert wanted[path] == sha(data)
        wanted[path] = sha(data)
    for path, expected in source['source_hashes'].items():
        add(Path(path), expected)
    for path, expected in rm['files'].items():
        add(Path(path), expected)
    # Only source_hashes and supplemental files are expanded; private opaque hashes stay references.
    extras = """INDEPENDENT_REVIEW_MANIFEST.json PREPARATION_STATUS.md PROPOSAL_HANDOFF.md""".split()
    for name in extras:add(RD/name)
    other = """current_state/ROUND07_PREPARATION_ACCEPTANCE.md current_state/COORDINATOR.md
current_state/monitor.md current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
current_state/history_baseline.md current_state/ROUND07_PREPARATION_DECISION.md
monitoring/CHECK_20261001T200037Z.json monitoring/SCHEDULED_ROUND07_ACCEPTED_20261001T2001.json
ops/ROUND06_COMPLETION_PUBLICATION_RECEIPT.json
ops/package_records.py ops/download_records.py ops/prepare_round07_publication.py
ops/publish_round07_preparation.py ops/ssh_proxy.py ops/FAILED_ROUND_ARCHIVE_POLICY.md
ops/round07_publication/README.md
ops/heartbeat_round07_preparation/BEFORE.toml ops/heartbeat_round07_preparation/AFTER.toml
ops/heartbeat_round07_preparation/UPDATE_ARGUMENTS.json
ops/heartbeat_round07_preparation/HEARTBEAT_ROUND07_PREPARATION.json""".split()
    for name in other:add(ROOT/name)
    entries = []
    mappings = []
    for path, expected in sorted(wanted.items(), key=lambda x:str(x[0])):
        data = read(path, expected)
        if path.is_relative_to(ROOT):
            dest = path.relative_to(ROOT).as_posix()
        else:
            assert path.is_relative_to(Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/frozen_reference'))
            dest = 'ops/round07_publication/external_sources/' + path.relative_to('/home/inspur/MTO-1').as_posix()
            save(ROOT/dest, data)
        entries.append(dict(path=dest, bytes=len(data), sha256=expected))
        mappings.append(dict(original_absolute_path=str(path), archive_path=dest, sha256=expected))
    save(OP/'RESTORE_MAPPING.json', encoded({'files':mappings,'instructions':'Restore bytes to original absolute paths in an isolated compatible environment. The full 185-source closure and nine supplement members are copied explicitly; restore bytes at original server paths. Source references to private_checkpoint_opaque_hashes, raw archives, indices or predictions never authorize payload copying. Private preflight states must not initialize production. Earlier full split/failure and deferred-QC records remain inherited from parent 68520927 and its ancestors. No production authority is included.'}))
    data=read(OP/'RESTORE_MAPPING.json')
    entries.append(dict(path='ops/round07_publication/RESTORE_MAPPING.json',bytes=len(data),sha256=sha(data)))
    assert len({e['path'] for e in entries}) == len(entries)
    assert sum(e['bytes'] for e in entries) < 25*1024*1024
    inventory=dict(prepared_at_utc=stamp(),parent='68520927333b27c48381401c8516a6709b79c40e',
        phase='accepted_preparation_no_production',source_manifest_sha256=SOURCE,independent_review_sha256=REVIEW,
        independent_review_manifest_sha256=REVIEW_MANIFEST,
        root_acceptance_sha256=ACCEPTANCE,approved_readme_sha256=README,files=entries,
        record_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),private_artifacts_excluded=True)
    save(OP/'ARCHIVE_INVENTORY.json',encoded(inventory))
    print(json.dumps({k:v for k,v in inventory.items() if k!='files'}))

def package(review_sha):
    bind()
    invbytes=read(OP/'ARCHIVE_INVENTORY.json'); inv=json.loads(invbytes)
    approval=json.loads(read(OP/'PUBLICATION_SOURCE_REVIEW.json',review_sha))
    assert approval['passed'] and approval['inventory_sha256']==sha(invbytes)
    assert approval['prepare_script_sha256']==sha(read(Path(__file__).resolve()))
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round07_preparation.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round07_publication/'+name,bytes=len(data),sha256=sha(data)))
    for entry in entries:
        path=PurePosixPath(entry['path'])
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
        data=inspect(ROOT/entry['path']);assert len(data)==entry['bytes'] and sha(data)==entry['sha256']
    manifest=dict(round_id='round07_congruence_preparation',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Reviewed exact 185-source closure plus nine supplemental review members and explicit operational text; no private arrays/tensors/credentials or filename exceptions.',
        files=entries,total_bytes=sum(e['bytes'] for e in entries))
    out=OP/'server_records.tar.gz';assert not out.exists()
    with tarfile.open(out,'w:gz') as tar:
        for entry in entries:
            data=read(ROOT/entry['path'],entry['sha256']);info=tarfile.TarInfo(entry['path']);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
        data=encoded(manifest);info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
    result=dict(bundle=str(out),sha256=sha(read(out)),files=len(entries),total_bytes=manifest['total_bytes'])
    save(OP/'SERVER_PACKAGE_RECEIPT.json',encoded(result));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--package-review-sha256');a=p.parse_args()
    assert a.prepare != bool(a.package_review_sha256)
    prepare() if a.prepare else package(a.package_review_sha256)
