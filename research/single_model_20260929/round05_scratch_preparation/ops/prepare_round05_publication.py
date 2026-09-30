#!/usr/bin/env python3
"""Exact lightweight Round05 archival closure; no scientific imports or execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD = ROOT / 'round05_scratch_preparation'
SD = ROOT / 'dataset_audit_20260930'
OP = ROOT / 'ops/round05_publication'
SOURCE = '66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1'
REVIEW = 'ef7d90250fd0491fba1432201416de983afa86ee6ae7ebf6e086735f5e10ff19'
REVIEW_MANIFEST = 'ca33d0dc331d0ba6caae70b1d4291a7d1584e0f290b1d10b99c6f54ec6411185'
SPLIT_MANIFEST = '54a4de262de180fa107611de9af90b54def2dcf908f9f0331d0b03ffa0076f6b'
EXCEPTION_REVIEW = 'd39165a88ea63e0c459ee89f367822e8f16fefe32fbc0b8a25c177a5d726b583'
ACCEPTANCE = '8fbd9c47e51cc3fcdda7af40737eecbe2065700fd9f6ba93c5205b9f49e6c6b8'
README = '22b615bff7f9bef544a4e55e44650533b153aaadf41a4b27d585473fe6d0fa49'
EXCEPTIONS = {str(SD / p): 'e69286aef0f19f1ec4dbb63102c3811ce7219101ab1fde84b85286a886ac4fec'
    for p in ('EXISTING_DATASET_AUDIT.json', 'repair01_original/EXISTING_DATASET_AUDIT.json')}

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
        assert EXCEPTIONS.get(str(path)) == sha(data), 'Forbidden artifact name: ' + str(path)
    assert len(data) <= 5 * 1024 * 1024
    text = data.decode('utf-8-sig')
    assert '\x00' not in text and not SECRET.search(text), str(path)
    assert path.suffix.lower() != '.svg', 'No SVG needed in this preparation archive'
    return data

def bind():
    source = json.loads(read(RD/'FROZEN_MANIFEST.json', SOURCE))
    review = json.loads(read(RD/'INDEPENDENT_PREPARATION_REVIEW.json', REVIEW))
    rm = json.loads(read(RD/'INDEPENDENT_REVIEW_MANIFEST.json', REVIEW_MANIFEST))
    sm = json.loads(read(SD/'FINAL_LIGHTWEIGHT_MANIFEST.json', SPLIT_MANIFEST))
    exception = json.loads(read(RD/'PUBLICATION_AGGREGATE_EXCEPTION_REVIEW.json', EXCEPTION_REVIEW))
    read(ROOT/'current_state/ROUND05_PREPARATION_ACCEPTANCE.md', ACCEPTANCE)
    read(OP/'README.md', README)
    assert review['passed'] and review['frozen_manifest_sha256'] == SOURCE
    assert not review['production_fit_authorized'] and len(source['source_hashes']) == 82
    assert review['source_hashes'] == source['source_hashes']
    assert rm['source_manifest_sha256'] == SOURCE and rm['independent_review_sha256'] == REVIEW
    assert sm['passed'] and sm['record_count'] == 89
    assert exception['passed'] and exception['exceptions'] == EXCEPTIONS
    return source, rm, sm

def prepare():
    source, rm, sm = bind()
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
    for entry in sm['files']:
        add(SD/entry['path'], entry['sha256'])
        assert (SD/entry['path']).stat().st_size == entry['bytes']
    for name in ('FINAL_LIGHTWEIGHT_MANIFEST.json', 'FINAL_LIGHTWEIGHT_ALLOWLIST.txt'):
        add(SD/name)
    snapshot = json.loads(read(RD/'preflight_history/gpu_attempt01/SNAPSHOT.json'))
    for entry in snapshot['records']:
        add(Path(entry['snapshot']), entry['sha256'])
    extras = '''INDEPENDENT_REVIEW_MANIFEST.json PREPARATION_STATUS.md INDEPENDENT_REVIEW_STATUS.md
TRAIN_READER_REVIEW.md BUFFER_ROUNDTRIP_DIAGNOSIS.log CPU_PREFLIGHT_01.log CPU_PREFLIGHT_02.log
FREEZE_01.log INDEPENDENT_GATE_CHECKS_01.log INDEPENDENT_READER_CHECKS_01.log READER_CHECKS_01.log
preserve_gpu_attempt01.py preflight_history/cpu_attempt01/CPU_PREFLIGHT_01.log
preflight_history/cpu_attempt01/cpu_preflight.py'''.split()
    for name in extras:
        add(RD/name)
    for attempt, terminal in [('statistics_attempt','COMPLETE.json'), ('gpu_preflight_attempt','FAILED.json'),
            ('gpu_identity_diagnostic_attempt','COMPLETE.json'), ('gpu_preflight_attempt_retry01','COMPLETE.json')]:
        for name in [terminal, 'LAUNCH_RECEIPT.json', 'MONITOR_REGISTRATION.json', 'REGISTRATION_TOOL.log', 'stage.log', 'status.json']:
            add(RD/'ops'/attempt/name)
        if attempt != 'statistics_attempt':
            add(RD/'ops'/attempt/'GPU_ADMISSION.xml')
    for name in ['statistics_wrapper.log','gpu_preflight_wrapper.log','gpu_identity_diagnostic_wrapper.log','gpu_preflight_wrapper_retry01.log']:
        add(RD/'ops'/name)
    other = '''current_state/ROUND05_PREPARATION_ACCEPTANCE.md current_state/COORDINATOR.md
current_state/monitor.md current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
current_state/history_baseline.md
current_state/ROUND05_PREPARATION_DECISION.md
deferred_qc/QC_DESIGN_RECOMMENDATION_20260930.md deferred_qc/qc_response_congruence/synthetic_preflight.py
deferred_qc/qc_response_congruence/SYNTHETIC_RESULTS.json deferred_qc/qc_response_congruence/SOURCE_RECEIPT.json
monitoring/CHECK_20260930T115345Z.json monitoring/SCHEDULED_PREPARATION_20260930T1153.json
ops/ROUND04_COMPLETION_PUBLICATION_RECEIPT.json ops/ROUND04_POSTPUBLICATION_VERIFICATION.json
ops/package_records.py ops/download_records.py ops/prepare_round05_publication.py
ops/publish_round05_preparation.py ops/ssh_proxy.py ops/FAILED_ROUND_ARCHIVE_POLICY.md
ops/round05_publication/README.md'''.split()
    for name in other:
        add(ROOT/name)
    entries = []
    mappings = []
    for path, expected in sorted(wanted.items(), key=lambda x:str(x[0])):
        data = read(path, expected)
        if path.is_relative_to(ROOT):
            dest = path.relative_to(ROOT).as_posix()
        else:
            assert path.is_relative_to(Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/frozen_reference'))
            dest = 'ops/round05_publication/external_sources/' + path.relative_to('/home/inspur/MTO-1').as_posix()
            save(ROOT/dest, data)
        entries.append(dict(path=dest, bytes=len(data), sha256=expected))
        mappings.append(dict(original_absolute_path=str(path), archive_path=dest, sha256=expected))
    save(OP/'RESTORE_MAPPING.json', encoded({'files':mappings,'instructions':'Restore bytes to original absolute paths in an isolated compatible environment. Raw data/partition and technical tensors remain private; their hashes are provenance only. No production authority is included. Deferred QC is untested model-design evidence.'}))
    data=read(OP/'RESTORE_MAPPING.json')
    entries.append(dict(path='ops/round05_publication/RESTORE_MAPPING.json',bytes=len(data),sha256=sha(data)))
    assert len({e['path'] for e in entries}) == len(entries)
    assert sum(e['bytes'] for e in entries) < 25*1024*1024
    inventory=dict(prepared_at_utc=stamp(),parent='8497e0ba10efc3226197c894f46507acdad8ec54',
        phase='accepted_preparation_no_production',source_manifest_sha256=SOURCE,independent_review_sha256=REVIEW,
        independent_review_manifest_sha256=REVIEW_MANIFEST,split_lightweight_manifest_sha256=SPLIT_MANIFEST,
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
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round05_preparation.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round05_publication/'+name,bytes=len(data),sha256=sha(data)))
    for entry in entries:
        path=PurePosixPath(entry['path'])
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
        data=inspect(ROOT/entry['path']);assert len(data)==entry['bytes'] and sha(data)==entry['sha256']
    manifest=dict(round_id='round05_scratch_preparation',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Reviewed exact text records and two hash-bound aggregate metadata exceptions; no private arrays/tensors/credentials.',
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
