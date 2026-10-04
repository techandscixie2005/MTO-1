from pathlib import Path
import hashlib,json,subprocess
W=Path('C:/Users/master/Documents/ChatGPT/MTO/research_state');S='/home/inspur/MTO-1/research/single_model_20260929'
code=r'''
from pathlib import Path
import hashlib,json,collections,datetime,subprocess
r=Path('/home/inspur/MTO-1/research/single_model_20260929')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'monitoring/CHECK_20261004T160955Z.json';o=json.loads(p.read_text());prior=json.loads((r/'monitoring/SCHEDULED_ROUND09_TERMINAL_20261004T0409.json').read_text());oldobs=json.loads((r/'monitoring/CHECK_20261004T040809Z.json').read_text())
assert len(o['runs'])==37 and not o['incidents'] and not o['interventions']
assert all(x['process_identity'] is None and not x['observed_gpu_uuids'] for x in o['runs'])
counts=dict(collections.Counter(x['state'] for x in o['runs']));assert counts=={'COMPLETE_MARKER_PRESENT':35,'FAILURE_MARKER_REVIEW_REQUIRED':2}
for x in o['runs']:
 if x['state']=='FAILURE_MARKER_REVIEW_REQUIRED':
  old=next(z for z in oldobs['runs'] if z['id']==x['id']);assert x['files']['failure_path'].get('content')==old['files']['failure_path'].get('content')
checks=[]
for old in prior['round09_runs']:
 x=next(v for v in o['runs'] if v['id']==old['id']);rd=r/x['run_dir'];t=x['files']['completion_path']['content']
 assert sha(rd/'FIT_COMPLETE.json')==old['terminal_sha256'] and sha(rd/'status.json')==old['status_sha256']
 assert sha(rd/'history.jsonl')==old['history_sha256']==t['history_sha256'] and sha(r/old['log_path'])==old['log_sha256']
 assert t['completed_epoch']==60 and t['steps']==112860 and t['test_evaluated'] is False
 assert all(v['exists'] for v in x['checkpoints'].values())
 for key, meta in x['checkpoints'].items():
  assert {k:v for k,v in meta.items() if k!='age_seconds'}=={k:v for k,v in old['checkpoint_file_metadata'][key].items() if k!='age_seconds'}
 assert True and (rd/'geometry_best.pt').stat().st_size==old['geometry_checkpoint_file_bytes']
 checks.append(dict(id=x['id'],registered_identity=old['registered_identity'],original_worker_absent=True,terminal_status_history_log_hashes_unchanged=True,completed_epoch=60,steps=112860,test_evaluated_declared=False,checkpoint_file_metadata_present=True))
cron=hashlib.sha256(subprocess.run(['crontab','-l'],capture_output=True,check=True).stdout).hexdigest();assert cron=='13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e'
pub=r/'ops/ROUND09_COMPLETION_PUBLICATION_RECEIPT.json';assert sha(pub)=='039677a3cdfaf5f895b1b8eb16a2e8b510462c1ae53503c07ea17c7ad16db8cd'
v=dict(schema_version=1,scope='Scheduled registered-owned identity/terminal/resource metadata only',scheduled_wakeup='2026-10-04T16:10Z',checked_at_utc=o['checked_at_utc'],receipt_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_record=p.name,canonical_sha256=sha(p),registry_sha256=o['registry_sha256'],registered_runs=37,state_counts=counts,all_registered_original_workers_absent=True,active_owned_jobs=0,incidents=[],interventions=[],historical_failures_unchanged=True,round09_checks=checks,gpu_snapshot=o['gpu_snapshot'],cron_sha256=cron,cron_unchanged=True,heartbeat_unchanged_by_check=True,no_duplicate_automation=True,no_tensor_dataset_decoding_or_scientific_inference=True,closed_published_commit='6e23b9d9c7eeb479b478ebb69ec061024de62455',publication_receipt_sha256=sha(pub),scope_now='Root-accepted Round10 preparation; publication in progress, production blocked')
target=r/'monitoring/SCHEDULED_ROUND10_ACCEPTED_20261004T1610.json';assert not target.exists();target.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(path=str(target),sha256=sha(target),canonical_record=p.name,canonical_sha256=sha(p),gpu_summary=[dict(index=g['index'],memory=g['memory_used'],utilization=g['utilization'],pids=[q['pid'] for q in g['compute_processes']]) for g in o['gpu_snapshot']['gpus']])))
'''
run=subprocess.run(['ssh','USTC-A800','/usr/bin/python3','-'],input=code.encode(),capture_output=True,timeout=45)
if run.returncode:raise RuntimeError(run.stderr.decode())
v=json.loads(run.stdout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in [(Path(v['path']).name,v['sha256']),(v['canonical_record'],v['canonical_sha256'])]:
 p=W/'single_model_20260929/monitoring'/name;subprocess.run(['scp','-q','USTC-A800:'+S+'/monitoring/'+name,str(p)],check=True,timeout=30);assert sha(p)==h

import tomllib
O=W/'single_model_20260929/ops';H=O/'heartbeat_round10_accepted';H.mkdir(exist_ok=True)
def mirror(p,rel):
 subprocess.run(['scp','-q',str(p),'USTC-A800:'+S+'/'+rel],check=True,timeout=30)
 assert subprocess.run(['ssh','USTC-A800','sha256sum '+S+'/'+rel],capture_output=True,check=True,timeout=30).stdout.decode().split()[0]==sha(p)
decision=W/'ROUND10_PREPARATION_ACCEPTANCE.md';dh=sha(decision);assert dh=='fcb45cfafff2678811cca8ee56c5dee68e0948378821703aaf8d25f7d2751a91';mirror(decision,'current_state/'+decision.name)
summary=f"""Round10 preparation is ACCEPTED by root ROUND10_PREPARATION_ACCEPTANCE.md SHA256 {dh}. Frozen458-path manifest7b4bed854c54eb9a1faabcb817b101c4bafe0f80869d7102242f1746b0c2aa2f, final review17124e86a1fa555821659d6d6ec2bf7775a2c812b9462eaffc8cea2f3019d4b8, distinct23-member preparation supplemente3b411f87eb6222c426020f73e4a62486c0bc3e5c4404db5f7d4b25e0398c1a2 and reportec5440d18c2a9f8104293406771fbcc02a01d31aed4b2541353d6e425780e53f are fixed. Original15-member proposal closure remains unchanged.

Seven CPU toy calls and exactly six discarded first128TRAIN updates completed once; no validation/TEST or efficacy result. Preserve the failed metadata freeze, four-file snapshot, narrow reviewed alias repair and distinct successful FREEZE_RECEIPT_02. This recovery repeated no numerical stage and changed no scientific setting. All sealed records remain immutable; technical states cannot initialize production.

Publisher QC/history now execute D-FIRST lightweight archival and exact README/inventory/staged-byte review, followed by atomic non-force main+research publication descending6e23b9d9c7eeb479b478ebb69ec061024de62455 and remote byteverification. Exclude weights, checkpoints, optimizer states, learned/raw/identity/split/prediction arrays, caches and credentials. No scientific repeat. Production remains blocked until verified publication and separate exact root execution authority under standing user authority; no additional user approval gate. Future fixedGPU1/2 admission is independent of technical selection; no fallback or blind restart.

Retain originalMTO/LE+Ls/WD0, fixed .001 versus .001 epochs1–30/.0003 epochs31–60, exact Adam moments/RNG/order continuity, dual+.003 gate and pre-drop checkpoint interpretation limit. No tuning, seed, extension, averaging or TEST access. RetainedRound05 .44716940136585204 remains; .60 unmet.

Due16:10Z observation at16:09:55Z: all37 registered original workers absent,35complete plus two unchanged historical failures, zero incidents/interventions. The technical fixture's normal completion is already observed; no new failure. Round09 terminal/status/history/log and stable checkpoint metadata remain unchanged. Canonical monitoring/{v['canonical_record']} SHA256 {v['canonical_sha256']}; concise monitoring/{Path(v['path']).name} SHA256 {v['sha256']}. Existing four-hour heartbeat and notification policy remain unchanged; no duplicate. Unrelated jobs and cron preserved.

"""
for name in ['COORDINATOR.md','monitor.md']:
 p=W/name;s=p.read_text(encoding='utf-8');backup=H/('BEFORE_'+name);assert not backup.exists();backup.write_bytes(p.read_bytes())
 start=s.index('Updated ');end=s.index('\n\n',start);s=s[:start]+'Updated 2026-10-04 after16:10Z monitoring. Round10 preparation ACCEPTED fcb45cfa; D-first publication in progress. No active owned worker; production blocked until separate authority. No scientific repeat; TEST sealed.'+s[end:]
 if name=='COORDINATOR.md':s=s.replace('## Immediate work and authority\n\n','## Immediate work and authority\n\n'+summary+'### Historical bounded Round10 preparation and earlier snapshots\n\n',1)
 else:s=s.replace('## Current Round10 bounded preparation','## Current Round10 accepted preparation\n\n'+summary+'## Historical Round10 bounded preparation',1)
 p.write_text(s,encoding='utf-8',newline='\n');mirror(p,'current_state/'+name)
a=Path('C:/Users/master/.codex/automations/qm9s-e-a/automation.toml');assert not (H/'BEFORE.toml').exists();(H/'BEFORE.toml').write_bytes(a.read_bytes());settings=tomllib.loads(a.read_text(encoding='utf-8'));prompt=settings['prompt'];start=prompt.index('Round10 bounded PREPARATION is authorized');end=prompt.index('\n\nDevelop quantum',start);prompt=prompt[:start]+summary+'A scheduled wakeup grants no additional execution authority.'+prompt[end:]
(H/'UPDATE_ARGUMENTS.json').write_text(json.dumps(dict(mode='update',id=settings['id'],kind=settings['kind'],name=settings['name'],prompt=prompt,status=settings['status'],rrule=settings['rrule'],targetThreadId=settings['target_thread_id']),indent=2)+'\n',encoding='utf-8')
mirror(Path(__file__),'ops/'+Path(__file__).name)
print(json.dumps(dict(receipt_sha256=v['sha256'],canonical_sha256=v['canonical_sha256'],decision_sha256=dh,coordinator_sha256=sha(W/'COORDINATOR.md'),monitor_sha256=sha(W/'monitor.md'),gpu_summary=v['gpu_summary'])))
