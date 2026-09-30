"""Seal explicit lightweight science records. Never copy tensors or raw arrays."""
import json,time
from pathlib import Path
from common import ROOT,CAMPAIGN,sha,read,atomic_json
output=ROOT/'completion/TERMINAL_MANIFEST.json'
assert not output.exists(),'Completion manifest is immutable'
assert read(ROOT/'completion/ANALYSIS_RECEIPT.json')['passed']
names=[
    'terminal_analysis.py','TERMINAL_ANALYSIS_CONFIG.json','write_completion_report.py','seal_completion.py',
    'inspect_first_epoch.py','FIRST_EPOCH_INSPECTION_NOTE.md','PRODUCTION_EXECUTION_AUTHORIZATION.json',
    'PRODUCTION_RUNTIME_HANDOFF.md','ops/FIRST_EPOCH_METADATA.json','ops/terminal_analysis_01.log',
    'completion/ROUND05_RESULTS.json','completion/ANALYSIS_RECEIPT.json',
    'completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg','completion/ROUND05_REPORT.md',
    'completion/ROUND05_REPRODUCE.md','completion/NEXT_DIRECTION_RECOMMENDATION.md']
paths=[ROOT/n for n in names]
for arm in ('control','adapter','decorrelation','both'):
    rd=ROOT/'runs'/arm
    terminal=read(rd/'FIT_COMPLETE.json')
    assert terminal['completed_epoch']==60 and terminal['steps']==112860 and not terminal['test_evaluated']
    assert not (rd/'FAILED.json').exists()
    paths += [rd/n for n in ('FIT_COMPLETE.json','BEST.json','DATA_ACCESS.json','history.jsonl','status.json')]
    attempts=list((rd/'attempts').iterdir());assert len(attempts)==1
    paths += [attempts[0]/n for n in ('LAUNCH_RECEIPT.json','GPU_ADMISSION.xml','REGISTRATION_TOOL.log','train.log')]
entries=[]
for path in paths:
    assert path.is_file() and not path.is_symlink()
    assert path.suffix.lower() in ('.json','.jsonl','.md','.py','.xml','.log','.svg')
    path.read_text()  # Reject nontext bytes; substantive content reviewed separately.
    entries.append({'path':str(path),'relative_path':str(path.relative_to(ROOT)),
                    'sha256':sha(path),'bytes':path.stat().st_size})
atomic_json({'format':'round05_lightweight_terminal_manifest_v1','created_unix':time.time(),
    'frozen_scientific_manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),
    'preparation_commit':'9de80e37e1586c2d3068b188ed2c242ee9883147',
    'analysis_receipt_sha256':sha(ROOT/'completion/ANALYSIS_RECEIPT.json'),
    'count':len(entries),'bytes':sum(x['bytes'] for x in entries),'files':entries,
    'inherited_sources':'Exact frozen82-file source closure and prior failure/preflight/split records are in the preparation publication; restore by its manifest and dependency mapping.',
    'independent_reviews_root_decision_and_current_monitor_records':'Publisher adds exact reviewed records separately; no circular binding.',
    'exclusions':['all checkpoint and optimizer tensors','raw or prediction arrays','split or identity arrays','raw datasets','caches','credentials','private preflight directories'],
    'test_evaluated':False,'completed_four_arm_fixed60':True},output)
print(json.dumps({'manifest_sha256':sha(output),'count':len(entries),'bytes':sum(x['bytes'] for x in entries)}))
