"""Seal explicit lightweight Round09 files; never expand private input hashes."""
import json,time
from common import ROOT,sha,read,atomic_json,require_cpu
require_cpu()
output=ROOT/'completion/TERMINAL_MANIFEST.json'
assert not output.exists(),'Completion manifest is immutable'
receipt=read(ROOT/'completion/ANALYSIS_RECEIPT.json')
assert receipt['passed'] and receipt['no_model_inference']
manifest_sha=sha(ROOT/'FROZEN_MANIFEST.json')
assert manifest_sha=='52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930'
assert receipt['manifest_sha256']==manifest_sha
names=[
 'terminal_analysis.py','TERMINAL_ANALYSIS_CONFIG.json','TERMINAL_ANALYSIS_SOURCE_REVIEW.json',
 'write_completion_report.py','seal_completion.py','inspect_first_epoch.py',
 'FIRST_EPOCH_INSPECTOR_SOURCE_REVIEW.json','PRODUCTION_EXECUTION_AUTHORIZATION.json',
 'CURRENT_RUNTIME_HANDOFF.md','ops/ROUND09_RUNTIME_HANDOFF.md',
 'ops/FIRST_EPOCH_METADATA.json','ops/FIRST_EPOCH_METADATA_01.log','ops/FIRST_EPOCH_METADATA_INVOCATION_01.json',
 'ops/PRODUCTION_GATE_01.json','ops/PRODUCTION_LAUNCH_01.log','ops/PRODUCTION_LAUNCH_01_RECEIPT.json',
 'ops/INDEPENDENT_PRODUCTION_BINDING_REVIEW.json',
 'ops/terminal_analysis_01.log','ops/TERMINAL_ANALYSIS_INVOCATION_01.json','ops/write_completion_report_01.log',
 'completion/ROUND09_RESULTS.json','completion/ANALYSIS_RECEIPT.json',
 'completion/ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg','completion/REGULARIZATION_TRAJECTORIES.svg',
 'completion/ROUND09_REPORT.md','completion/ROUND09_REPRODUCE.md',
]
paths=[ROOT/name for name in names]
for arm in ('zero_decay','coupled_l2'):
 run=ROOT/'runs'/arm;terminal=read(run/'FIT_COMPLETE.json')
 assert terminal['completed_epoch']==60 and terminal['steps']==112860
 assert terminal['manifest_sha256']==manifest_sha and terminal['test_evaluated'] is False
 assert not (run/'FAILED.json').exists()
 paths += [run/name for name in ('FIT_COMPLETE.json','BEST.json','DATA_ACCESS.json','history.jsonl','status.json')]
 attempts=list((run/'attempts').iterdir());assert len(attempts)==1 and attempts[0].is_dir()
 paths += [attempts[0]/name for name in ('LAUNCH_RECEIPT.json','GPU_ADMISSION.xml','REGISTRATION_TOOL.log','train.log')]
assert len(paths)==44 and len(paths)==len(set(paths))
entries=[]
for path in paths:
 assert path.is_file() and not path.is_symlink()
 assert path.suffix.lower() in ('.json','.jsonl','.md','.py','.xml','.log','.svg')
 path.read_text(encoding='utf-8')
 entries.append({'path':str(path),'relative_path':str(path.relative_to(ROOT)),
                 'sha256':sha(path),'bytes':path.stat().st_size})
atomic_json({
 'format':'round09_lightweight_terminal_manifest_v1','created_unix':time.time(),
 'frozen_scientific_manifest_sha256':manifest_sha,'preparation_commit':'566aa5e02c2f7cad6dc8c7b8c732ad45dfb9c8be',
 'analysis_receipt_sha256':sha(ROOT/'completion/ANALYSIS_RECEIPT.json'),
 'count':len(entries),'bytes':sum(v['bytes'] for v in entries),'files':entries,
 'files_are_exact_archive_payload_allowlist':True,'private_input_hashes_are_provenance_only':True,
 'inherited_sources':'Exact360-file preparation closure, preserved resource failure/retry, split and preflight records are inherited from preparation publication; restore by its manifest and dependency map.',
 'independent_reviews_root_decision_and_current_monitor_records':'Publisher binds exact reviewed records separately, avoiding circular references.',
 'exclusions':['checkpoints/optimizer/learned tensors','raw or prediction arrays','split/identity arrays','raw datasets','caches','credentials','private preflight directories'],
 'test_evaluated':False,'completed_two_arm_fixed60':True,
},output)
print(json.dumps({'manifest_sha256':sha(output),'count':len(entries),'bytes':sum(v['bytes'] for v in entries)}))
