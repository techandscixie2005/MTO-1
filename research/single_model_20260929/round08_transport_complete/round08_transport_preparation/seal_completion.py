"""Seal explicit lightweight Round08 records; private references are not payloads."""
import json,time
from common import ROOT,sha,read,atomic_json,require_cpu
require_cpu()
output=ROOT/'completion/TERMINAL_MANIFEST.json'
assert not output.exists(),'Completion manifest is immutable'
receipt=read(ROOT/'completion/ANALYSIS_RECEIPT.json')
assert receipt['passed'] and receipt['no_model_inference']
manifest_sha=sha(ROOT/'FROZEN_MANIFEST.json')
assert manifest_sha=='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146'
assert receipt['manifest_sha256']==manifest_sha
names=[
    'terminal_analysis.py','TERMINAL_ANALYSIS_CONFIG.json','TERMINAL_ANALYSIS_SOURCE_REVIEW.json',
    'write_completion_report.py','seal_completion.py','inspect_first_epoch.py',
    'FIRST_EPOCH_INSPECTOR_SOURCE_REVIEW.json','PRODUCTION_EXECUTION_AUTHORIZATION.json',
    'CURRENT_RUNTIME_HANDOFF.md','ops/ROUND08_RUNTIME_HANDOFF.md',
    'ops/FIRST_EPOCH_METADATA.json','ops/first_epoch_inspection_01.log','ops/production_launch_01.log',
    'ops/terminal_analysis_01.log','ops/write_completion_report_01.log',
    'completion/ROUND08_RESULTS.json','completion/ANALYSIS_RECEIPT.json',
    'completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg','completion/TRANSPORT_TRAJECTORIES.svg',
    'completion/ROUND08_REPORT.md','completion/ROUND08_REPRODUCE.md','completion/NEXT_RESEARCH_QUESTION.md',
    'completion/REPORT_WORDING_AMENDMENT.json','completion/report_wording_history/write_completion_report_01.py',
    'completion/report_wording_history/ROUND08_REPORT_01.md',
]
paths=[ROOT/name for name in names]
for arm in ('original','local','neighbor'):
    run=ROOT/'runs'/arm;terminal=read(run/'FIT_COMPLETE.json')
    assert terminal['completed_epoch']==60 and terminal['steps']==112860
    assert terminal['manifest_sha256']==manifest_sha
    assert terminal['test_evaluated'] is False and not (run/'FAILED.json').exists()
    paths += [run/name for name in ('FIT_COMPLETE.json','BEST.json','DATA_ACCESS.json','history.jsonl','status.json')]
    attempts=list((run/'attempts').iterdir());assert len(attempts)==1 and attempts[0].is_dir()
    paths += [attempts[0]/name for name in ('LAUNCH_RECEIPT.json','GPU_ADMISSION.xml','REGISTRATION_TOOL.log','train.log')]
assert len(paths)==len(set(paths))
entries=[]
for path in paths:
    assert path.is_file() and not path.is_symlink()
    assert path.suffix.lower() in ('.json','.jsonl','.md','.py','.xml','.log','.svg')
    path.read_text(encoding='utf-8')
    entries.append({'path':str(path),'relative_path':str(path.relative_to(ROOT)),
                    'sha256':sha(path),'bytes':path.stat().st_size})
atomic_json({
    'format':'round08_lightweight_terminal_manifest_v1','created_unix':time.time(),
    'frozen_scientific_manifest_sha256':manifest_sha,
    'preparation_commit':'a10eb9a35b26014f00138c61476b33e9070bb69b',
    'analysis_receipt_sha256':sha(ROOT/'completion/ANALYSIS_RECEIPT.json'),
    'count':len(entries),'bytes':sum(v['bytes'] for v in entries),'files':entries,
    'files_are_exact_archive_payload_allowlist':True,'private_input_hashes_are_provenance_only':True,
    'inherited_sources':'Exact258-file preparation closure, prior failures, split and preflight records are inherited from preparation publication; restore by its manifest and dependency map.',
    'independent_reviews_root_decision_and_current_monitor_records':'Publisher adds exact reviewed records separately; no circular binding.',
    'exclusions':['all checkpoints/gate/optimizer tensors','raw or prediction arrays','split or identity arrays',
                  'raw datasets','caches','credentials','private preflight directories'],
    'test_evaluated':False,'completed_three_arm_fixed60':True,
},output)
print(json.dumps({'manifest_sha256':sha(output),'count':len(entries),'bytes':sum(v['bytes'] for v in entries)}))
