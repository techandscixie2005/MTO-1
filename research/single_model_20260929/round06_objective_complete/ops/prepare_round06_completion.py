#!/usr/bin/env python3
"""Reviewed Round06 completion text closure; never copy private input-hash references."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from monitor import ROOT, stamp
from package_records import EXTENSIONS, DENIED_PARTS, SECRET

RD=ROOT/'round06_objective_preparation'
CD=RD/'completion'
OP=ROOT/'ops/round06_completion_publication'
PARENT='c26a860b972a1c01267f8b766393772173515b92'
SOURCE='4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b'
TERMINAL='e8b9f9f0f6d270423791710c1de9f77c9732bac1bf31c3df26afd8e6a4897469'
REVIEW='e11df595b17e268f5ca0b67394c601dfbc4ada32ba7b72f267586e0b4c12188d'
CLOSEOUT='1d3b802ec0ec68a751bdd0fdf8ca2a6b212675320d420a759357bfbb2b867d01'
DECISION='8f1b4e6de20024c904c90f551a677b74ac56d5b213581c07672a979b0fe03a39'
README='9bf62f48eaed4c69811c672b65320511374e4948776dfb298e2ff729a1c3faf0'
SVG=CD/'ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg'
SVG_HASH='31482c7f43e48033fbc988a381d39b35c4cf2c68fb5e953447c77ac337e41dc0'

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
        assert path==SVG and sha(data)==SVG_HASH
        assert not re.search(r'<(?:script|foreignObject)\b|\bon\w+\s*=|javascript:|(?:xlink:)?href\s*=\s*["\x27](?:https?:|data:)',text,re.I)
    return data
def bind():
    read(RD/'FROZEN_MANIFEST.json',SOURCE)
    tm=json.loads(read(CD/'TERMINAL_MANIFEST.json',TERMINAL))
    review=json.loads(read(CD/'INDEPENDENT_TERMINAL_REVIEW.json',REVIEW))
    cm=json.loads(read(CD/'INDEPENDENT_CLOSEOUT_MANIFEST.json',CLOSEOUT))
    read(ROOT/'current_state/ROUND06_DECISION.md',DECISION);read(OP/'README.md',README)
    assert tm['completed_two_arm_fixed60'] and not tm['test_evaluated']
    assert tm['frozen_scientific_manifest_sha256']==SOURCE and tm['preparation_commit']==PARENT
    assert tm['count']==39 and len(tm['files'])==39
    assert review['passed'] and review['frozen_manifest_sha256']==SOURCE and not review['test_evaluated']
    assert review['analysis_receipt_sha256']==sha(read(CD/'ANALYSIS_RECEIPT.json'))
    assert review['analysis_result_sha256']==sha(read(CD/'ROUND06_RESULTS.json'))
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
    # Only the explicit 39 public files are copied. input_hashes are private provenance.
    for e in tm['files']:
        path=Path(e['path']);assert path==RD/e['relative_path']
        add(path,e['sha256'],e['bytes'])
    assert sum(e['bytes'] for e in tm['files'])==tm['bytes']
    for name,expected in cm['files'].items():add(Path(name),expected)
    extras='''round06_objective_preparation/completion/TERMINAL_MANIFEST.json
round06_objective_preparation/completion/INDEPENDENT_CLOSEOUT_MANIFEST.json
round06_objective_preparation/FROZEN_MANIFEST.json round06_objective_preparation/ops/seal_completion_01.log
current_state/COORDINATOR.md current_state/monitor.md current_state/history_baseline.md
current_state/science_implementation.md current_state/RESEARCH_RESUMPTION_20260930.md
ops/ROUND06_PREPARATION_PUBLICATION_RECEIPT.json
monitoring/CHECK_20261001T075723Z.json monitoring/SCHEDULED_ROUND06_TERMINAL_20261001T0759.json
monitoring/ROUND06_STARTUP_CONFIRMATION_20261001T0358.json ops/check_round06_startup.py
ops/heartbeat_round06_active/HEARTBEAT_ROUND06_ACTIVE.json
ops/heartbeat_round06_active/BEFORE.toml ops/heartbeat_round06_active/AFTER.toml
ops/package_records.py ops/download_records.py ops/monitor.py ops/ssh_proxy.py
ops/prepare_round06_completion.py ops/publish_round06_completion.py
ops/FAILED_ROUND_ARCHIVE_POLICY.md ops/round06_completion_publication/README.md'''.split()
    for name in extras:add(ROOT/name)
    restore=OP/'RESTORE.md'
    save(restore,('''# Round06 completion restoration

This archive supplements immutable preparation commit '''+PARENT+'''. Restore preparation files from research/single_model_20260929/round06_objective_preparation/ using its ops/round06_publication/RESTORE_MAPPING.json. That captures all 134 frozen sources and 17 external sources, reviewed preflight, original missing-mirror event and independent metadata-check repair. Earlier split and Round05 failure records remain in inherited ancestry. Preserve their original absolute server layout and pinned environment.

Then overlay the completion archive's campaign-relative text files under /home/inspur/MTO-1/research/single_model_20260929. The README snapshot belongs at repository README.md. Scientific source manifest remains '''+SOURCE+'''. The final report/reproduction instructions distinguish already completed fits, metadata audits and saved-output analysis. Never repeat completed stages as an archival check.

The 39-entry TERMINAL_MANIFEST.files is the public science allowlist. ANALYSIS_RECEIPT.input_hashes and terminal checkpoint/prediction paths are private provenance only: do not resolve those references into archive members. Models, optimizer states, prediction/split/identity arrays and raw data stay private server artifacts. Review and root decision authorize completion publication only, not new training or TEST scoring. Next-direction text authorizes proposal development only after publication, without implementation or fits. The earlier algebraic QC design remains in inherited archives.
''').encode())
    add(restore)
    entries=[names[k] for k in sorted(names)];assert sum(e['bytes'] for e in entries)<25*1024*1024
    inv=dict(prepared_at_utc=stamp(),phase='completed_round06_reviewed_publication',parent=PARENT,
        frozen_manifest_sha256=SOURCE,terminal_manifest_sha256=TERMINAL,independent_terminal_review_sha256=REVIEW,
        independent_closeout_manifest_sha256=CLOSEOUT,root_decision_sha256=DECISION,approved_readme_sha256=README,
        files=entries,record_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),private_inputs_not_copied=True)
    save(OP/'ARCHIVE_INVENTORY.json',encoded(inv));print(json.dumps({k:v for k,v in inv.items() if k!='files'}))
def package(review_sha):
    bind();invbytes=read(OP/'ARCHIVE_INVENTORY.json');inv=json.loads(invbytes)
    approval=json.loads(read(OP/'PUBLICATION_SOURCE_REVIEW.json',review_sha))
    assert approval['passed'] and approval['inventory_sha256']==sha(invbytes)
    assert approval['prepare_script_sha256']==sha(read(Path(__file__).resolve()))
    assert approval['publisher_script_sha256']==sha(read(ROOT/'ops/publish_round06_completion.py'))
    entries=list(inv['files'])
    for name in ['ARCHIVE_INVENTORY.json','PUBLICATION_SOURCE_REVIEW.json']:
        data=inspect(OP/name);entries.append(dict(path='ops/round06_completion_publication/'+name,bytes=len(data),sha256=sha(data)))
    for e in entries:
        p=PurePosixPath(e['path']);assert not p.is_absolute() and '..' not in p.parts and '\\' not in str(p)
        data=inspect(ROOT/e['path']);assert len(data)==e['bytes'] and sha(data)==e['sha256']
    manifest=dict(round_id='round06_objective_complete',packaged_at_utc=stamp(),source_root=str(ROOT),
        policy='Exact reviewed text-only closeout; one hash-bound aggregate SVG; no arrays/tensors/raw data/credentials.',files=entries,total_bytes=sum(e['bytes'] for e in entries))
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
