#!/usr/bin/env python3
"""Reviewed Round08 completion text closure; never copy private input-hash references."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD=ROOT/'round08_transport_preparation'
CD=RD/'completion'
OP=ROOT/'ops/round08_completion_publication'
PARENT='a10eb9a35b26014f00138c61476b33e9070bb69b'
SOURCE='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146'
TERMINAL='b5db6f2ec8dc11226f44f420f435766f60503a97f4f4cf4260af63bff127e76a'
REVIEW='21279ddfb34f7f73c496814c7a9960f9aac4ebeb092059de1f0df725c6b60ac9'
CLOSEOUT='b794c348b90cd459786580c6298819d0b84491882f1d0e78521f6354a7f2debf'
DECISION='1099239c16237f85c185b02ebbb24b2c41f7812a10e4c018d5c35a5a68d3028d'
README='0dddf4d2450a4fcff751c6ad354d1afc24a7921b73e3c88b343020ec145a2f98'
SVG_HASHES={'/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation/completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg': '7a1d4c077dae98f6f1af520881bed8b02888d3680d3c12336ab3d19e37530cdf', '/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation/completion/TRANSPORT_TRAJECTORIES.svg': '26837119e8fabf0f4804ad658c087bc8124b88f062bd75bfcfd1ebdd4908df45'}


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path, expected=None):
    assert path.is_file() and not path.is_symlink() and path.resolve()==path, str(path)
    data=path.read_bytes()
    if expected: assert sha(data)==expected, str(path)
    return data
def encoded(obj): return (json.dumps(obj,indent=2)+'\n').encode()
def save(path,data):
    assert not path.exists(), 'Preserve existing operational artifact: '+str(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def inspect(path):
    assert path.is_relative_to(ROOT),str(path)
    data=read(path);assert path.suffix.lower() in EXTENSIONS
    assert not set(path.parts)&DENIED_PARTS
    assert not any(x.startswith(('private_partition','private_preflight')) for x in path.parts)
    assert not re.search(r'(?:prediction|dataset|credential|optimizer_state|model_weight)',path.name,re.I),str(path)
    assert len(data)<=5*1024*1024
    text=data.decode('utf-8-sig');assert '\x00' not in text and not SECRET.search(text),str(path)
    if path.suffix.lower()=='.svg':
        assert str(path) in SVG_HASHES and sha(data)==SVG_HASHES[str(path)]
        assert not re.search(r'<(?:script|foreignObject)\b|\bon\w+\s*=|javascript:|(?:xlink:)?href\s*=\s*["\x27](?:https?:|data:)',text,re.I)
    return data
def bind():
    read(RD/'FROZEN_MANIFEST.json',SOURCE)
    tm=json.loads(read(CD/'TERMINAL_MANIFEST.json',TERMINAL))
    review=json.loads(read(CD/'INDEPENDENT_TERMINAL_REVIEW.json',REVIEW))
    cm=json.loads(read(CD/'INDEPENDENT_CLOSEOUT_MANIFEST.json',CLOSEOUT))
    read(ROOT/'current_state/ROUND08_DECISION.md',DECISION);read(OP/'README.md',README)
    assert tm['completed_three_arm_fixed60'] and not tm['test_evaluated']
    assert tm['frozen_scientific_manifest_sha256']==SOURCE and tm['preparation_commit']==PARENT
    assert tm['count']==52 and len(tm['files'])==52
    assert review['passed'] and review['frozen_manifest_sha256']==SOURCE and not review['test_evaluated']
    assert review['analysis_receipt_sha256']==sha(read(CD/'ANALYSIS_RECEIPT.json'))
    assert review['analysis_result_sha256']==sha(read(CD/'ROUND08_RESULTS.json'))
    assert cm['passed'] and cm['source_manifest_sha256']==SOURCE and cm['terminal_manifest_sha256']==TERMINAL
    assert cm['independent_review_sha256']==REVIEW and cm['root_decision_sha256']==DECISION
    assert cm['preparation_parent_commit']==PARENT and not cm['new_scientific_execution_authorized']
    return tm,cm
def prepare():
    tm,cm=bind();assert not (OP/'ARCHIVE_INVENTORY.json').exists()
    names={}
    def add(path,expected=None,size=None):
        data=inspect(path)
        if expected:assert sha(data)==expected,str(path)
        if size is not None:assert len(data)==size,str(path)
        name=path.relative_to(ROOT).as_posix()
        if name in names:assert names[name]['sha256']==sha(data)
        names[name]=dict(path=name,bytes=len(data),sha256=sha(data))
    # Only the explicit 52 public files are copied. input_hashes are private provenance.
    for e in tm['files']:
        path=Path(e['path']);assert path==RD/e['relative_path']
        add(path,e['sha256'],e['bytes'])
    assert sum(e['bytes'] for e in tm['files'])==tm['bytes']
    for name,expected in cm['files'].items():add(Path(name),expected)
    extras='''round08_transport_preparation/completion/TERMINAL_MANIFEST.json
round08_transport_preparation/completion/INDEPENDENT_CLOSEOUT_MANIFEST.json
round08_transport_preparation/FROZEN_MANIFEST.json round08_transport_preparation/ops/seal_completion_01.log
current_state/COORDINATOR.md current_state/monitor.md current_state/history_baseline.md
current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
ops/ROUND08_PREPARATION_PUBLICATION_RECEIPT.json
monitoring/CHECK_20261003T040200Z.json monitoring/SCHEDULED_ROUND08_TERMINAL_20261003T0403.json
monitoring/CHECK_20261003T040002Z.json
ops/close_monitor_round08_20261003T0403.py
monitoring/ROUND08_STARTUP_CONFIRMATION_20261003T0002.json
monitoring/CHECK_20261003T000442Z.json ops/check_round08_startup.py
ops/heartbeat_round08_active/HEARTBEAT_ROUND08_ACTIVE.json
ops/heartbeat_round08_active/BEFORE.toml ops/heartbeat_round08_active/AFTER.toml
ops/heartbeat_round08_active/UPDATE_ARGUMENTS.json
ops/package_records.py ops/download_records.py ops/monitor.py ops/ssh_proxy.py
ops/prepare_round08_completion.py ops/publish_round08_completion.py
ops/FAILED_ROUND_ARCHIVE_POLICY.md ops/round08_completion_publication/README.md'''.split()
    for name in extras:add(ROOT/name)
    restore=OP/'RESTORE.md'
    save(restore,('''# Round08 completion restoration

This archive supplements immutable preparation commit '''+PARENT+'''. Restore preparation files from research/single_model_20260929/round08_transport_preparation/ using its ops/round08_publication/RESTORE_MAPPING.json. That captures all 258 frozen source/dependency/audit records and 17 external sources, reviewed nine-update technical preflight and its exact reviews, the read-only audit/proposal history, registration barrier correction, CPU shell-exit evidence and reviewer state-comparison correction. Earlier split repairs and Round05/Round06 failure records remain in inherited ancestry. Preserve their original absolute server layout and pinned environment.

Then overlay the completion archive's campaign-relative text files under /home/inspur/MTO-1/research/single_model_20260929. The README snapshot belongs at repository README.md. Scientific source manifest remains '''+SOURCE+'''. The final report/reproduction instructions distinguish already completed fits, metadata audits and saved-output analysis. Never repeat completed stages as an archival check.

The 52-entry TERMINAL_MANIFEST.files is the public science allowlist. ANALYSIS_RECEIPT.input_hashes and terminal checkpoint/prediction paths are private provenance only: do not resolve those references into archive members. Models, optimizer states, prediction/split/identity arrays and raw data stay private server artifacts. Review and root decision authorize completion publication only, not new training or TEST scoring. Root authorizes only one prespecified regularization proposal on unchanged original MTO after publication, using existing source and aggregate history. Late validation deterioration is not proof of overfitting. Specify optimizer/decay semantics, parameter groups, one a-priori strength, zero-decay control, fresh initialization, fixed budget and retained-reference allocation gate. Independent proposal review precedes any preparation decision; no new target decoding, model inference, implementation, synthetic optimizer updates, fits, coefficient search or TEST access. The earlier algebraic QC design remains in inherited archives.
''').encode())
    add(restore)
    entries=[names[k] for k in sorted(names)];assert sum(e['bytes'] for e in entries)<25*1024*1024
    inv=dict(prepared_at_utc=stamp(),phase='completed_round08_reviewed_publication',parent=PARENT,
        frozen_manifest_sha256=SOURCE,terminal_manifest_sha256=TERMINAL,independent_terminal_review_sha256=REVIEW,
        independent_closeout_manifest_sha256=CLOSEOUT,root_decision_sha256=DECISION,approved_readme_sha256=README,
        files=entries,record_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),private_inputs_not_copied=True)
    save(OP/'ARCHIVE_INVENTORY.json',encoded(inv));print(json.dumps({k:v for k,v in inv.items() if k!='files'}))
def package(review_sha):
    bind();invbytes=read(OP/'ARCHIVE_INVENTORY.json');inv=json.loads(invbytes)
    approval=json.loads(read(OP/'PUBLICATION_SOURCE_REVIEW.json',review_sha))
    assert approval['passed'] and approval['inventory_sha256']==sha(invbytes)
    assert approval['prepare_script_sha256']==sha(read(Path(__file__).resolve()))
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round08_completion.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round08_completion_publication/'+name,bytes=len(data),sha256=sha(data)))
    for e in entries:
        p=PurePosixPath(e['path']);assert not p.is_absolute() and '..' not in p.parts and '\\' not in str(p)
        data=inspect(ROOT/e['path']);assert len(data)==e['bytes'] and sha(data)==e['sha256']
    manifest=dict(round_id='round08_transport_complete',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Exact reviewed text-only closeout; two explicitly hash-bound aggregate SVGs; no arrays/tensors/raw data/credentials.',files=entries,total_bytes=sum(e['bytes'] for e in entries))
    out=OP/'server_records.tar.gz';assert not out.exists()
    with tarfile.open(out,'w:gz') as tar:
        for e in entries:
            data=read(ROOT/e['path'],e['sha256']);info=tarfile.TarInfo(e['path']);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
        data=encoded(manifest);info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
    result=dict(bundle=str(out),sha256=sha(read(out)),files=len(entries),total_bytes=manifest['total_bytes'])
    save(OP/'SERVER_PACKAGE_RECEIPT.json',encoded(result));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--package-review-sha256');a=p.parse_args()
    assert a.prepare != bool(a.package_review_sha256)
    prepare() if a.prepare else package(a.package_review_sha256)
