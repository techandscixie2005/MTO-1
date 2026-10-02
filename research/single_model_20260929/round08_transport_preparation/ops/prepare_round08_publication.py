#!/usr/bin/env python3
"""Exact lightweight Round08 archival closure; no scientific imports or execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD = ROOT / 'round08_transport_preparation'
OP = ROOT / 'ops/round08_publication'
SOURCE = 'de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146'
REVIEW = '87de2d873f2a24033a2f646ac825f26abbbae7eadbd5305a723fbb6a17093e8b'
REVIEW_MANIFEST = '4ee5e4d010e0e4c689562adcea2b0e6b1a4d07dff70b4067e0bb8178f039f99d'
ACCEPTANCE = '17efd5b82d201401c5d18cdf40bc76bb318be7799289481beaf16e501eb3bf61'
README = 'd60f899dc4e9ae7d3fd34f9b2686599eb1d86e3ba2f4b97f883f18d53a1330c8'
EXACT_NUMERIC_METADATA = {
    RD/'ops/cpu_synthetic_01.pid': '6361ef35651f2204f1c13123f9afbcd84c01980072811c7ef8c3b67f1cc1114f',
    RD/'ops/cpu_synthetic_01.exit': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
}


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
    if path in EXACT_NUMERIC_METADATA:
        assert sha(data)==EXACT_NUMERIC_METADATA[path] and len(data)<=32
        assert re.fullmatch(rb'[0-9]+\n',data), str(path)
    else:
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
    read(ROOT/'current_state/ROUND08_PREPARATION_ACCEPTANCE.md', ACCEPTANCE)
    read(OP/'README.md', README)
    assert review['passed'] and review['frozen_manifest_sha256'] == SOURCE
    assert not review['production_authorized'] and len(source['source_hashes']) == 258
    assert source['actual_discarded_optimizer_updates']==9 and source['preparation_only_no_production_authorization']
    assert review['source_hashes'] == source['source_hashes']
    assert rm['source_manifest_sha256'] == SOURCE and rm['independent_review_sha256'] == REVIEW
    assert rm['phase']=='preparation_only' and not rm['production_execution_authorized']
    assert len(rm['files'])==9
    assert review['report_sha256']==sha(read(RD/'PREPARATION_REPORT.md'))
    audit=json.loads(read(ROOT/'bottleneck_audit_20261002/INDEPENDENT_REVIEW_MANIFEST.json','5ba90e8fe798b9cb4f05d26f7065d669aa17e93025bc87d6a97dca7a687131b6'))
    assert len(audit['files'])==10 and audit['input_reference_hashes_are_not_archive_payload_allowlists']
    for relative,entry in audit['files'].items():
        path=ROOT/relative
        assert str(path) in source['source_hashes'] and source['source_hashes'][str(path)]==entry['sha256']
        assert len(read(path,entry['sha256']))==entry['bytes']
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
    extras = """INDEPENDENT_REVIEW_MANIFEST.json PREPARATION_STATUS.md INDEPENDENT_REVIEW_STATUS.md RUNNER_HANDOFF.md""".split()
    for name in extras:add(RD/name)
    other = """current_state/ROUND08_PREPARATION_ACCEPTANCE.md current_state/COORDINATOR.md
current_state/monitor.md current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
current_state/history_baseline.md current_state/ROUND08_PREPARATION_DECISION.md
monitoring/CHECK_20261002T200136Z.json monitoring/SCHEDULED_ROUND08_ACCEPTED_20261002T2002.json
ops/ROUND07_COMPLETION_PUBLICATION_RECEIPT.json
ops/package_records.py ops/download_records.py ops/prepare_round08_publication.py
ops/publish_round08_preparation.py ops/ssh_proxy.py ops/FAILED_ROUND_ARCHIVE_POLICY.md
ops/round08_publication/README.md
ops/heartbeat_round08_preparation/BEFORE.toml ops/heartbeat_round08_preparation/AFTER.toml
ops/heartbeat_round08_preparation/UPDATE_ARGUMENTS.json
ops/heartbeat_round08_preparation/HEARTBEAT_ROUND08_PREPARATION.json""".split()
    for name in other:add(ROOT/name)
    entries = []
    mappings = []
    for path, expected in sorted(wanted.items(), key=lambda x:str(x[0])):
        data = read(path, expected)
        if path.is_relative_to(ROOT):
            dest = path.relative_to(ROOT).as_posix()
        else:
            assert path.is_relative_to(Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/frozen_reference'))
            dest = 'ops/round08_publication/external_sources/' + path.relative_to('/home/inspur/MTO-1').as_posix()
            save(ROOT/dest, data)
        entries.append(dict(path=dest, bytes=len(data), sha256=expected))
        mappings.append(dict(original_absolute_path=str(path), archive_path=dest, sha256=expected))
    save(OP/'RESTORE_MAPPING.json', encoded({'files':mappings,'instructions':'Restore bytes to original absolute paths in an isolated compatible environment. The full 258-source closure, nine supplemental members and all ten audit/proposal/history closure members (already within the source closure) are copied explicitly; restore bytes at original server paths. Source references to private_checkpoint_opaque_hashes, raw archives, indices or predictions never authorize payload copying. Private preflight states must not initialize production. Earlier full split/failure and deferred-QC records remain inherited from parent 703c2cd7 and its ancestors. No production authority is included.'}))
    data=read(OP/'RESTORE_MAPPING.json')
    entries.append(dict(path='ops/round08_publication/RESTORE_MAPPING.json',bytes=len(data),sha256=sha(data)))
    assert len({e['path'] for e in entries}) == len(entries)
    assert sum(e['bytes'] for e in entries) < 25*1024*1024
    inventory=dict(prepared_at_utc=stamp(),parent='703c2cd7c68f4068160ab9d3dbbab64d4d5af6ba',
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
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round08_preparation.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round08_publication/'+name,bytes=len(data),sha256=sha(data)))
    for entry in entries:
        path=PurePosixPath(entry['path'])
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
        data=inspect(ROOT/entry['path']);assert len(data)==entry['bytes'] and sha(data)==entry['sha256']
    manifest=dict(round_id='round08_transport_preparation',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Reviewed exact 258-source closure plus nine supplemental review members and explicit operational text; only two exact path/SHA-bound numeric PID/exit text extension exceptions; no private arrays/tensors/credentials.',
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
