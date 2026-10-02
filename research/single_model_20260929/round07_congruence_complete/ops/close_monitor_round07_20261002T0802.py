"""Close one scheduled metadata observation; never decode scientific payloads."""
from pathlib import Path
import hashlib,json,subprocess
W=Path('C:/Users/master/Documents/ChatGPT/MTO/research_state');O=W/'single_model_20260929/ops';S='/home/inspur/MTO-1/research/single_model_20260929'
code=r'''
from pathlib import Path
import hashlib,json,collections,datetime,subprocess
r=Path('/home/inspur/MTO-1/research/single_model_20260929')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'monitoring/CHECK_20261002T080125Z.json';o=json.loads(p.read_text());reg=json.loads((r/'ops/registry.json').read_text());start=json.loads((r/'monitoring/CHECK_20261002T000721Z.json').read_text())
assert len(o['runs'])==29 and not o['incidents'] and not o['interventions']
assert all(x['process_identity'] is None and not x['observed_gpu_uuids'] for x in o['runs'])
counts=dict(collections.Counter(x['state'] for x in o['runs']));assert counts=={'COMPLETE_MARKER_PRESENT':27,'FAILURE_MARKER_REVIEW_REQUIRED':2}
oldfail={x['id']:x['files']['failure_path'].get('content') for x in start['runs'] if x['state']=='FAILURE_MARKER_REVIEW_REQUIRED'}
for x in o['runs']:
 if x['id'] in oldfail:assert x['files']['failure_path'].get('content')==oldfail[x['id']]
prior=[]
for name in ['CHECK_20261002T040001Z.json','CHECK_20261002T080002Z.json']:
 q=r/'monitoring'/name;v=json.loads(q.read_text());prior.append((q,v))
checks=[]
for arm in ['original','scalar','tensor']:
 x=next(x for x in o['runs'] if x['run_dir']=='round07_congruence_preparation/runs/'+arm);rd=r/x['run_dir'];t=x['files']['completion_path']['content'];registered=next(z for z in reg['runs'] if z['id']==x['id'])
 assert x['state']=='COMPLETE_MARKER_PRESENT' and not x['files']['failure_path']['exists']
 assert t['completed_epoch']==60 and t['steps']==112860 and t['test_evaluated'] is False
 assert t['manifest_sha256']=='68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203'
 assert sha(rd/'history.jsonl')==t['history_sha256']
 assert all(z['exists'] and z['bytes']>0 for z in x['checkpoints'].values()) and (rd/'geometry_best.pt').is_file()
 for q,v in prior:
  old=next(z for z in v['runs'] if z['id']==x['id']);assert old['state']=='COMPLETE_MARKER_PRESENT' and old['process_identity'] is None
  assert old['files']['completion_path']['content']==t and old['files']['status_path']['content']==x['files']['status_path']['content']
  assert not old['files']['failure_path']['exists']
 logpath=registered['log_path']
 checks.append(dict(id=x['id'],arm=arm,registered_identity=registered['identity'],original_worker_absent=True,gpu_uuid=registered['gpu_uuid'],observed_gpu_uuids=x['observed_gpu_uuids'],terminal_sha256=sha(rd/'FIT_COMPLETE.json'),status_sha256=sha(rd/'status.json'),history_sha256=sha(rd/'history.jsonl'),log_path=logpath,log_sha256=sha(r/logpath),completed_epoch=60,steps=112860,test_evaluated_declared=False,checkpoint_file_metadata=x['checkpoints'],geometry_checkpoint_file_bytes=(rd/'geometry_best.pt').stat().st_size,terminal_already_observed_at_0400Z=True,terminal_and_status_content_unchanged_since_0400Z=True,scientific_checkpoint_content_not_verified_by_monitor=True))
cron=hashlib.sha256(subprocess.run(['crontab','-l'],capture_output=True,check=True).stdout).hexdigest();assert cron=='13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e'
v=dict(schema_version=1,scope='Scheduled registered-owned process/terminal/resource metadata only',scheduled_wakeup='2026-10-02T08:02Z',checked_at_utc=o['checked_at_utc'],receipt_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_record=p.name,canonical_sha256=sha(p),registry_sha256=o['registry_sha256'],registered_runs=29,state_counts=counts,all_registered_original_workers_absent=True,active_owned_jobs=0,incidents=[],interventions=[],historical_failures_unchanged=True,round07_runs=checks,existing_cron_records=[dict(name=q.name,sha256=sha(q),checked_at_utc=z['checked_at_utc'],incidents=z['incidents'],interventions=z['interventions']) for q,z in prior],missing_app_0402_observation_not_training_failure=True,normal_completion_transition_already_recorded_by_cron=True,gpu_snapshot=o['gpu_snapshot'],cron_sha256=cron,cron_unchanged=True,heartbeat_unchanged_by_check=True,no_duplicate_automation=True,no_checkpoint_dataset_decoding_or_scientific_inference=True,terminal_scientific_review_and_root_decision_pending=True,publication_on_hold=True)
target=r/'monitoring/SCHEDULED_ROUND07_TERMINAL_20261002T0802.json';assert not target.exists();target.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(path=str(target),sha256=sha(target),canonical_record=p.name,canonical_sha256=sha(p),cron_records=v['existing_cron_records'],gpu_summary=[dict(index=g['index'],memory=g['memory_used'],utilization=g['utilization'],pids=[q['pid'] for q in g['compute_processes']]) for g in o['gpu_snapshot']['gpus']])))
'''
run=subprocess.run(['ssh','USTC-A800','/usr/bin/python3','-'],input=code.encode(),capture_output=True,timeout=45)
if run.returncode:raise RuntimeError(run.stderr.decode())
v=json.loads(run.stdout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
items=[(Path(v['path']).name,v['sha256']),(v['canonical_record'],v['canonical_sha256'])]+[(x['name'],x['sha256']) for x in v['cron_records']]
for name,h in items:
 p=W/'single_model_20260929/monitoring'/name;subprocess.run(['scp','-q','USTC-A800:'+S+'/monitoring/'+name,str(p)],check=True,timeout=30);assert sha(p)==h
summary='''Round07 original/scalar/tensor production is COMPLETE by terminal metadata: each60epochs/112860updates, TEST-unevaluated declaration, no failure marker, last/best/geometry checkpoint file metadata present. All29 registered original workers are absent:27 complete plus two unchanged historical failures. Zero new incidents/interventions. Science owns final checkpoint/source/order/access verification and saved-validation analysis; history owns independent review. Root scientific closeout decision and reviewed final manifests remain pending. Publisher holds D-first closeout until that decision. No restart, extension, extra seed, inference or TEST scoring is authorized.

The due08:02Z check observed state at08:01:25Z. Existing cron records CHECK_20261002T040001Z.json and CHECK_20261002T080002Z.json already recorded normal Round07 completion; terminal/status content remains identical. A missing app04:02 wake observation was not a training failure and requires no recovery. Terminal/history/status/log text hashes and checkpoint file metadata are recorded without decoding tensors or datasets. Canonical monitoring/'''+v['canonical_record']+''' SHA256 '''+v['canonical_sha256']+'''; concise monitoring/'''+Path(v['path']).name+''' SHA256 '''+v['sha256']+'''. Cron hash13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e is unchanged; this check changes no heartbeat/schedule and creates no duplicate. Unrelated jobs/resources remain untouched. Current verified preparation publication271d61f6 and retained v2 reference.447169401 remain; target.60 unmet, scientific results await independent review.

'''
for name in ['COORDINATOR.md','monitor.md']:
 p=W/name;s=p.read_text(encoding='utf-8');b=O/('BEFORE_20261002T0802_'+name);assert not b.exists();b.write_bytes(p.read_bytes())
 if name=='COORDINATOR.md':
  start=s.index('Updated ');end=s.index('\n\n## Immediate work',start);s=s[:start]+'Updated 2026-10-02 after08:02Z monitoring. Round07 terminal metadata shows allthree60-epoch fits complete; no active owned worker. Scientific terminal review/root decision and D-first closeout are pending. No restart or additional fit authority.'+s[end:]
  s=s.replace('## Immediate work and authority\n\n','## Immediate work and authority\n\n'+summary+'### Historical Round07 launch authority\n\n',1)
  s=s.replace('Current priority is safe completion and scheduled monitoring of the exact three authorized Round07 fits above;','Current priority is Round07 terminal scientific/independent review, root decision and later D-first closeout;')
 else:s=s.replace('## Current Round07 active production','## Current Round07 terminal review pending\n\n'+summary+'## Historical Round07 active production',1)
 p.write_text(s,encoding='utf-8',newline='\n');subprocess.run(['scp','-q',str(p),'USTC-A800:'+S+'/current_state/'+name],check=True,timeout=30);assert subprocess.run(['ssh','USTC-A800','sha256sum '+S+'/current_state/'+name],capture_output=True,check=True,timeout=30).stdout.decode().split()[0]==sha(p)
print(json.dumps(dict(receipt_sha256=v['sha256'],canonical_sha256=v['canonical_sha256'],coordinator_sha256=sha(W/'COORDINATOR.md'),monitor_sha256=sha(W/'monitor.md'),gpu_summary=v['gpu_summary'])))
