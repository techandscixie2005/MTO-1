#!/usr/bin/env python3
"""Reviewed Round09 completion text closure; never copy private input-hash references."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD=ROOT/'round09_regularization_preparation'
CD=RD/'completion'
OP=ROOT/'ops/round09_completion_publication'
PARENT='566aa5e02c2f7cad6dc8c7b8c732ad45dfb9c8be'
SOURCE='52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930'
TERMINAL='55ddb3957d4b1b11ea4cc10bf21d1ae6edfd3df52f74454893aae14e78cdb9bf'
REVIEW='a376eac31eeb940d96713561320aef32a23aa769b31c8f966aa4782754b872bc'
CLOSEOUT='e66c3f3b7798ded1a628e692358428a0ae093947f4941060b2e2438120ce17a5'
DECISION='8e09e19499c221c4d1cc03527ab4459e6bd8af6a29ee5090b2220e0b19515f7e'
README='0a38cffb47e776abc2876c5bd857bbb6c155d4cc2358769bb808214898667867'
SVG_HASHES={'/home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation/completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg': '759ee332c167781a8a14f1f934cfd6acbc87ad2e30b617dac5bfa3bfe8618bd0', '/home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation/completion/REGULARIZATION_TRAJECTORIES.svg': 'debeecf3446501c7fcfb39e712845a27b7a5672c20a65a922d265bd1f5079bd3'}


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
    read(ROOT/'current_state/ROUND09_DECISION.md',DECISION);read(OP/'README.md',README)
    assert tm['completed_two_arm_fixed60'] and not tm['test_evaluated']
    assert tm['frozen_scientific_manifest_sha256']==SOURCE and tm['preparation_commit']==PARENT
    assert tm['count']==44 and len(tm['files'])==44
    assert review['passed'] and review['source_manifest_sha256']==SOURCE and not review['test_access']
    assert review['analysis_receipt_sha256']==sha(read(CD/'ANALYSIS_RECEIPT.json'))
    assert review['results_sha256']==sha(read(CD/'ROUND09_RESULTS.json'))
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
    # Only the explicit 44 public files are copied. input_hashes are private provenance.
    for e in tm['files']:
        path=Path(e['path']);assert path==RD/e['relative_path']
        add(path,e['sha256'],e['bytes'])
    assert sum(e['bytes'] for e in tm['files'])==tm['bytes']
    for name,expected in cm['files'].items():add(Path(name),expected)
    extras='''round09_regularization_preparation/completion/TERMINAL_MANIFEST.json
round09_regularization_preparation/completion/INDEPENDENT_CLOSEOUT_MANIFEST.json
round09_regularization_preparation/FROZEN_MANIFEST.json round09_regularization_preparation/ops/seal_completion_01.log
round09_regularization_preparation/ops/independent_closeout_01.log
round09_regularization_preparation/ops/independent_closeout_02.log
round09_regularization_preparation/ops/INDEPENDENT_CLOSEOUT_BINDING_NOTE.json
current_state/COORDINATOR.md current_state/monitor.md current_state/history_baseline.md
current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
ops/ROUND09_PREPARATION_PUBLICATION_RECEIPT.json
monitoring/CHECK_20261004T040809Z.json monitoring/SCHEDULED_ROUND09_TERMINAL_20261004T0409.json
monitoring/CHECK_20261004T040002Z.json
ops/close_monitor_round09_20261004T0409.py
current_state/ROUND09_PRODUCTION_DECISION.md
round09_regularization_preparation/ops/INDEPENDENT_PRODUCTION_BINDING_REVIEW.json
monitoring/ROUND09_STARTUP_CONFIRMATION_20261004T0009.json
monitoring/CHECK_20261004T001218Z.json ops/check_round09_startup.py
ops/heartbeat_round09_active/HEARTBEAT_ROUND09_ACTIVE.json
ops/heartbeat_round09_active/BEFORE.toml ops/heartbeat_round09_active/AFTER.toml
ops/heartbeat_round09_active/UPDATE_ARGUMENTS.json
ops/package_records.py ops/download_records.py ops/monitor.py ops/ssh_proxy.py
ops/prepare_round09_completion.py ops/publish_round09_completion.py
ops/FAILED_ROUND_ARCHIVE_POLICY.md ops/round09_completion_publication/README.md'''.split()
    for name in extras:add(ROOT/name)
    restore=OP/'RESTORE.md'
    save(restore,('''# Round09 completion restoration

This archive supplements immutable preparation commit '''+PARENT+'''. Restore preparation files from research/single_model_20260929/round09_regularization_preparation/ using its ops/round09_publication/RESTORE_MAPPING.json. That captures all 360 frozen source/dependency/audit records and 20 exact external text mappings, reviewed six-update technical preflight, original zero-update GPU1 admission rejection, separate GPU4 retry, fixed parameter roster, immutable proposal review and sixteen-member final preparation supplement, plus earlier inherited audit/failure histories. Earlier split repairs and Round05/Round06 failure records remain in inherited ancestry. Preserve their original absolute server layout and pinned environment.

Then overlay the completion archive's campaign-relative text files under /home/inspur/MTO-1/research/single_model_20260929. The README snapshot belongs at repository README.md. Scientific source manifest remains '''+SOURCE+'''. The final report/reproduction instructions distinguish already completed fits, metadata audits and saved-output analysis. Never repeat completed stages as an archival check.

The 44-entry TERMINAL_MANIFEST.files is the public science allowlist. ANALYSIS_RECEIPT.input_hashes and terminal checkpoint/prediction paths are private provenance only: do not resolve those references into archive members. Models, optimizer states, prediction/split/identity arrays and raw data stay private server artifacts. Review and root decision authorize completion publication only, not new training or TEST scoring. Root authorizes read-only proposal development for one fixed learning-rate schedule contrast on unchanged original MTO with zero decay after publication, using existing source/aggregate trajectories and previous schedule trials. Audit the matched rationale; late validation decline does not prove optimization failure. Specify one schedule, contemporaneous fixed-LR control, initialization/order, budget, checkpoint rule, reference gate and seed-confirmation boundary. Independent proposal review and separate root preparation decision precede implementation or numerical work. No search, new split, target decoding, inference, extra updates, fit or TEST release. The earlier algebraic QC design remains in inherited archives.
''').encode())
    add(restore)
    entries=[names[k] for k in sorted(names)];assert sum(e['bytes'] for e in entries)<25*1024*1024
    inv=dict(prepared_at_utc=stamp(),phase='completed_round09_reviewed_publication',parent=PARENT,
        frozen_manifest_sha256=SOURCE,terminal_manifest_sha256=TERMINAL,independent_terminal_review_sha256=REVIEW,
        independent_closeout_manifest_sha256=CLOSEOUT,root_decision_sha256=DECISION,approved_readme_sha256=README,
        files=entries,record_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),private_inputs_not_copied=True)
    save(OP/'ARCHIVE_INVENTORY.json',encoded(inv));print(json.dumps({k:v for k,v in inv.items() if k!='files'}))
def package(review_sha):
    bind();invbytes=read(OP/'ARCHIVE_INVENTORY.json');inv=json.loads(invbytes)
    approval=json.loads(read(OP/'PUBLICATION_SOURCE_REVIEW.json',review_sha))
    assert approval['passed'] and approval['inventory_sha256']==sha(invbytes)
    assert approval['prepare_script_sha256']==sha(read(Path(__file__).resolve()))
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round09_completion.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round09_completion_publication/'+name,bytes=len(data),sha256=sha(data)))
    for e in entries:
        p=PurePosixPath(e['path']);assert not p.is_absolute() and '..' not in p.parts and '\\' not in str(p)
        data=inspect(ROOT/e['path']);assert len(data)==e['bytes'] and sha(data)==e['sha256']
    manifest=dict(round_id='round09_regularization_complete',packaged_at_utc=stamp(),source_root=str(ROOT),
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
