from pathlib import Path
import hashlib,json,subprocess,tomllib,datetime
W=Path('C:/Users/master/Documents/ChatGPT/MTO/research_state');O=W/'single_model_20260929/ops';H=O/'heartbeat_round10_accepted';S='/home/inspur/MTO-1/research/single_model_20260929';A=Path('C:/Users/master/.codex/automations')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=tomllib.loads((H/'BEFORE.toml').read_text(encoding='utf-8'));a=A/'qm9s-e-a/automation.toml';v=tomllib.loads(a.read_text(encoding='utf-8'));args=json.loads((H/'UPDATE_ARGUMENTS.json').read_text(encoding='utf-8'))
assert v['prompt']==args['prompt'] and v['updated_at']>b['updated_at']
for k in set(b)|set(v):
 if k not in ['prompt','updated_at']:assert b.get(k)==v.get(k),k
assert v['status']=='ACTIVE' and v['rrule']=='FREQ=HOURLY;INTERVAL=4'
matches=[]
for p in A.glob('*/automation.toml'):
 x=tomllib.loads(p.read_text(encoding='utf-8'))
 if x.get('status')=='ACTIVE' and x.get('kind')=='heartbeat' and x.get('target_thread_id')==v['target_thread_id'] and 'MTO' in x.get('prompt',''):matches.append(x['id'])
assert matches==['qm9s-e-a'];(H/'AFTER.toml').write_bytes(a.read_bytes());receipt=H/'HEARTBEAT_ROUND10_ACCEPTED.json';assert not receipt.exists()
receipt.write_text(json.dumps(dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,id=v['id'],status=v['status'],rrule=v['rrule'],target_thread_id=v['target_thread_id'],only_prompt_and_updated_at_changed=True,updated_using_app_tool=True,matching_active_heartbeat_count=1,before_sha256=sha(H/'BEFORE.toml'),after_sha256=sha(H/'AFTER.toml'),root_preparation_acceptance_sha256='fcb45cfafff2678811cca8ee56c5dee68e0948378821703aaf8d25f7d2751a91',monitor_receipt_sha256=sha(W/'single_model_20260929/monitoring/SCHEDULED_ROUND10_ACCEPTED_20261004T1610.json'),production_authorized=False,completed_discarded_train_updates=6,publication_in_progress=True,no_repeat_numerical_stages=True,no_extension_seed_or_test_authority=True),indent=2)+'\n',encoding='utf-8')
for name in ['COORDINATOR.md','monitor.md']:
 p=W/name;s=p.read_text(encoding='utf-8');needle='Existing four-hour heartbeat and notification policy remain unchanged; no duplicate.';assert needle in s;s=s.replace(needle,needle+' Verified receipt `ops/heartbeat_round10_accepted/HEARTBEAT_ROUND10_ACCEPTED.json` SHA256 `'+sha(receipt)+'`.',1);p.write_text(s,encoding='utf-8',newline='\n')
subprocess.run(['ssh','USTC-A800','mkdir -p '+S+'/ops/heartbeat_round10_accepted'],check=True,timeout=30)
files=[(H/n,'ops/heartbeat_round10_accepted/'+n) for n in ['BEFORE.toml','AFTER.toml','UPDATE_ARGUMENTS.json',receipt.name]]+[(W/n,'current_state/'+n) for n in ['COORDINATOR.md','monitor.md']]+[(O/n,'ops/'+n) for n in ['prepare_round10_accepted_handoff.py','verify_round10_accepted_heartbeat.py']]
for p,rel in files:
 subprocess.run(['scp','-q',str(p),'USTC-A800:'+S+'/'+rel],check=True,timeout=30)
 assert subprocess.run(['ssh','USTC-A800','sha256sum '+S+'/'+rel],capture_output=True,check=True,timeout=30).stdout.decode().split()[0]==sha(p)
print(json.dumps(dict(heartbeat_sha256=sha(receipt),coordinator_sha256=sha(W/'COORDINATOR.md'),monitor_sha256=sha(W/'monitor.md'))))
