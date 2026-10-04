#!/usr/bin/env python3
"""Exact lightweight Round10 archival closure; no scientific imports or execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD = ROOT / 'round10_schedule_preparation'
OP = ROOT / 'ops/round10_publication'
SOURCE = '7b4bed854c54eb9a1faabcb817b101c4bafe0f80869d7102242f1746b0c2aa2f'
REVIEW = '17124e86a1fa555821659d6d6ec2bf7775a2c812b9462eaffc8cea2f3019d4b8'
REVIEW_MANIFEST = 'e3b411f87eb6222c426020f73e4a62486c0bc3e5c4404db5f7d4b25e0398c1a2'
ACCEPTANCE = 'fcb45cfafff2678811cca8ee56c5dee68e0948378821703aaf8d25f7d2751a91'
README = '60c9ced4c8a9a540d57e988f7df3d29dc042a6a79c77c80f71bb558c41338f48'
EXACT_NUMERIC_METADATA = {
    ROOT/'round08_transport_preparation/ops/cpu_synthetic_01.pid': '6361ef35651f2204f1c13123f9afbcd84c01980072811c7ef8c3b67f1cc1114f',
    ROOT/'round08_transport_preparation/ops/cpu_synthetic_01.exit': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
}
EXACT_EXTERNAL_TEXT={'/home/inspur/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/nn/utils/clip_grad.py': 'a0cc48b76d4f3efb7a4266eb8e4c25a5dfad4972b609a9d56c4c3fe58afe28c9', '/home/inspur/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/optim/adam.py': 'bed22e29e5525e91329616ed8b9267afb77217e6eb1e5ac61e6ffff5d8384758', '/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/trainer.py': '78b9f6739540fd38b11cdbd216b5d3394db7b1cac6d653fe37e115630b7dc966', '/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/pyvenv.cfg': 'a4383b41b2b61afd0d1b76c26c48a32d7924913483682a2bce8026e2e341aad5', '/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/train_ea.py': '7c7b4d49db539566abd8c76da6fe3b7eabf96eaac3ac495997c5f074a49085dd', '/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/PROTOCOL.md': '9be3d271f86afac91a2b20957b4edee86bad6a14b0bd9058c564254ad8b151b2', '/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/config.json': '9ecb34590a64e389edbc455dbd22293593ce6c6e01c7d10b90b9cb5f6ec35048'}
EXTERNAL_DESTINATIONS={'/home/inspur/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/nn/utils/clip_grad.py': 'ops/round10_publication/external_sources/host/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/nn/utils/clip_grad.py', '/home/inspur/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/optim/adam.py': 'ops/round10_publication/external_sources/host/Documents/SpecGPT/envs/specgpt/lib/python3.10/site-packages/torch/optim/adam.py', '/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/trainer.py': 'ops/round10_publication/external_sources/host/MTO-1/experiments/qm9s_eta_Ef_20260926/trainer.py', '/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/pyvenv.cfg': 'ops/round10_publication/external_sources/host/MTO-1/experiments/qm9s_full_EA_20260925/env/pyvenv.cfg', '/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/train_ea.py': 'ops/round10_publication/external_sources/host/MTO-1/experiments/qm9s_full_EA_20260925/train_ea.py', '/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/PROTOCOL.md': 'ops/round10_publication/external_sources/host/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/PROTOCOL.md', '/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/config.json': 'ops/round10_publication/external_sources/host/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/config.json'}
EXACT_CONFIG_METADATA={Path('/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/pyvenv.cfg'): 'a4383b41b2b61afd0d1b76c26c48a32d7924913483682a2bce8026e2e341aad5', ROOT/'ops/round10_publication/external_sources/host/MTO-1/experiments/qm9s_full_EA_20260925/env/pyvenv.cfg': 'a4383b41b2b61afd0d1b76c26c48a32d7924913483682a2bce8026e2e341aad5'}



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
    elif path in EXACT_CONFIG_METADATA:
        assert sha(data)==EXACT_CONFIG_METADATA[path] and len(data)<=4096
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
    rm = json.loads(read(RD/'INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json', REVIEW_MANIFEST))
    read(ROOT/'current_state/ROUND10_PREPARATION_ACCEPTANCE.md', ACCEPTANCE)
    read(OP/'README.md', README)
    assert review['passed'] and review['frozen_manifest_sha256'] == SOURCE
    assert not review['production_authorized'] and len(source['source_hashes']) == 458
    assert source['actual_discarded_optimizer_updates']==6 and source['preparation_only_no_production_authorization']
    assert review['source_hashes'] == source['source_hashes']
    assert rm['source_manifest_sha256'] == SOURCE and rm['independent_review_sha256'] == REVIEW
    assert rm['phase']=='independent_round10_preparation_review_publication_supplement' and not rm['production_authorized']
    assert len(rm['files'])==23
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
    extras = """INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json INDEPENDENT_REVIEW_MANIFEST.json PREPARATION_STATUS.md INDEPENDENT_REVIEW_STATUS.md RUNNER_HANDOFF.md""".split()
    for name in extras:add(RD/name)
    other = """current_state/ROUND10_PREPARATION_ACCEPTANCE.md current_state/COORDINATOR.md
current_state/monitor.md current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
current_state/history_baseline.md current_state/ROUND10_PREPARATION_DECISION.md
monitoring/CHECK_20261004T160955Z.json monitoring/SCHEDULED_ROUND10_ACCEPTED_20261004T1610.json
monitoring/CHECK_20261004T121015Z.json monitoring/SCHEDULED_ROUND10_PREPARATION_20261004T1210.json
ops/ROUND09_COMPLETION_PUBLICATION_RECEIPT.json
ops/prepare_round10_accepted_handoff.py ops/verify_round10_accepted_heartbeat.py
ops/package_records.py ops/download_records.py ops/prepare_round10_publication.py
ops/publish_round10_preparation.py ops/ssh_proxy.py ops/FAILED_ROUND_ARCHIVE_POLICY.md
ops/round10_publication/README.md
ops/heartbeat_round10_accepted/BEFORE.toml ops/heartbeat_round10_accepted/AFTER.toml
ops/heartbeat_round10_accepted/UPDATE_ARGUMENTS.json
ops/heartbeat_round10_accepted/HEARTBEAT_ROUND10_ACCEPTED.json""".split()
    for name in other:add(ROOT/name)
    entries = []
    mappings = []
    for path, expected in sorted(wanted.items(), key=lambda x:str(x[0])):
        data = read(path, expected)
        if path.is_relative_to(ROOT):
            dest = path.relative_to(ROOT).as_posix()
        else:
            if str(path) in EXACT_EXTERNAL_TEXT:
                assert expected==EXACT_EXTERNAL_TEXT[str(path)]
                dest=EXTERNAL_DESTINATIONS[str(path)]
            else:
                assert path.is_relative_to(Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/frozen_reference'))
                dest = 'ops/round10_publication/external_sources/' + path.relative_to('/home/inspur/MTO-1').as_posix()
            save(ROOT/dest, data)
        entries.append(dict(path=dest, bytes=len(data), sha256=expected))
        mappings.append(dict(original_absolute_path=str(path), archive_path=dest, sha256=expected))
    save(OP/'RESTORE_MAPPING.json', encoded({'files':mappings,'instructions':'Restore bytes to original absolute paths in an isolated compatible environment. The full 458-source closure, twenty-three distinct final supplemental members and all ten audit/proposal/history closure members (already within the source closure) are copied explicitly; restore bytes at original server paths. Source references to private_checkpoint_opaque_hashes, raw archives, indices or predictions never authorize payload copying. Private preflight states must not initialize production. Earlier full split/failure and deferred-QC records remain inherited from parent 6e23b9d9 and its ancestors. No production authority is included.'}))
    data=read(OP/'RESTORE_MAPPING.json')
    entries.append(dict(path='ops/round10_publication/RESTORE_MAPPING.json',bytes=len(data),sha256=sha(data)))
    assert len({e['path'] for e in entries}) == len(entries)
    assert sum(e['bytes'] for e in entries) < 25*1024*1024
    inventory=dict(prepared_at_utc=stamp(),parent='6e23b9d9c7eeb479b478ebb69ec061024de62455',
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
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round10_preparation.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round10_publication/'+name,bytes=len(data),sha256=sha(data)))
    for entry in entries:
        path=PurePosixPath(entry['path'])
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
        data=inspect(ROOT/entry['path']);assert len(data)==entry['bytes'] and sha(data)==entry['sha256']
    manifest=dict(round_id='round10_schedule_preparation',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Reviewed exact 458-source closure plus twenty-three final supplemental review members and explicit operational text; only two inherited exact path/SHA-bound numeric PID/exit exceptions and one exact path/SHA-bound pyvenv.cfg text exception; twenty-four explicit external text restoration copies; no private arrays/tensors/credentials.',
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
