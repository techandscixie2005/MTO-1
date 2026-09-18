"""One-time recovery of a failed engineering gate; no scientific config edits."""
import datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from protocol import current

assert json.loads((ROOT/'results/preflight_regression_tests.json').read_text())['passed']
old=json.loads((ROOT/'jobs/submission.json').read_text())
assert old['stages']['1k']['job_id']=='1290122'
assert not list((ROOT/'runs').glob('*/experiment.json')),'Formal training exists; review before changing protocol'
assert not (ROOT/'results/preflight.json').exists()
frozen=json.loads((ROOT/'configs/frozen.json').read_text())
now=current()
changed={p for p in set(now)|set(frozen['files']) if now.get(p)!=frozen['files'].get(p)}
assert changed=={'scripts/preflight.py','tests/test_preflight.py'},changed
failed=subprocess.check_output(['sacct','-j','1290122','-X','-n','-o','State','--parsable2'],text=True)
assert 'FAILED' in failed,failed
for scale in ('10k','full'):
    job=old['stages'][scale]['job_id']
    state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
    assert state=='PENDING',(job,state)
    subprocess.run(['scancel',job],check=True)
archive=ROOT/'jobs/submission_failed_preflight_1290122.json'
assert not archive.exists()
(ROOT/'jobs/submission.json').replace(archive)
(ROOT/'configs/frozen.json').replace(ROOT/'reference/frozen_before_empty_buffer_fix.json')
subprocess.run([sys.executable,str(ROOT/'scripts/protocol.py'),'--freeze'],check=True,stdout=subprocess.DEVNULL)
new=json.loads((ROOT/'configs/frozen.json').read_text())
record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'failed_job':'1290122','failure':'Diagnostic max() reduction on a legitimate empty e3nn buffer',
        'changed_files':sorted(changed),'model_and_training_config_unchanged':True,
        'formal_fits_started':0,'old_protocol_id':frozen['protocol_id'],'new_protocol_id':new['protocol_id']}
(ROOT/'jobs/preflight_recovery_20260915.json').write_text(json.dumps(record,indent=2))
subprocess.run([sys.executable,str(ROOT/'scripts/submit_ordered.py')],check=True)
print(json.dumps(record,indent=2))
