"""Operational encoding repair; frozen scientific code/checkpoints stay unchanged."""
import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921')
os.chdir(root)
audit=root/'audit/utf8_recovery_20260921'
audit.mkdir(exist_ok=True)
if not (audit/'partial_1k_report').exists():shutil.copytree(root/'reports/1k',audit/'partial_1k_report')
if not (audit/'submission_before.json').exists():shutil.copy2(root/'jobs/submission.json',audit/'submission_before.json')
checks={}
for job in ('1299668','1299670'):
    state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
    if not state:continue
    description=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    assert state=='PENDING' and 'JobName=MTO11_report ' in description
    assert f'WorkDir={root} ' in description
    checks[job]=description
    subprocess.run(['scancel',job],check=True)
env=dict(os.environ,PYTHONUTF8='1',SBATCH_EXPORT='ALL,PYTHONUTF8=1')
(audit/'repair.json').write_text(json.dumps(dict(reason='report UnicodeEncodeError under ASCII locale',
    action='replace only queued report jobs; reuse all training jobs and checkpoints; enable PYTHONUTF8',
    cancelled_pending_reports=checks,environment={'PYTHONUTF8':'1','SBATCH_EXPORT':'ALL,PYTHONUTF8=1'}),indent=2),encoding='utf-8')
subprocess.run([sys.executable,'-u','submit.py'],env=env,check=True)
