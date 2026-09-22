"""Idempotent submission/resume. Reuses active jobs and repairs only this chain."""
import fcntl
import json
import subprocess
from pathlib import Path
from data_protocol import ROOT,SCALES,verify_data

def call(*args):return subprocess.check_output(args,text=True).strip()

def complete(kind,scale):
    from workflow import check_fits,check_stage
    if kind=='prepare':
        if not (ROOT/'data/PREPARED.json').exists():return False
        verify_data();return True
    if kind=='train':
        paths=[ROOT/f'runs/{scale}/seed_11/{v}/FIT_COMPLETE.json' for v in ('detanet_original_uv','detanet_mto_planned')]
        if not all(p.exists() for p in paths):return False
        check_fits(scale);return True
    if not (ROOT/f'reports/{scale}/STAGE_COMPLETE.json').exists():return False
    check_stage(scale)
    return scale!='full' or (ROOT/'reports/WORKFLOW_COMPLETE.json').exists()

def main():
    path=ROOT/'jobs/submission.json'
    with (ROOT/'jobs/submit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else dict(workspace=str(ROOT),nodes={},history=[])
        previous=None
        for kind,scale in [('prepare',None)]+[(k,s) for s in SCALES for k in ('train','report')]:
            key=kind if scale is None else f'{scale}_{kind}'
            if complete(kind,scale):
                print(key,'artifacts verified complete');previous=None;continue
            old=state['nodes'].get(key);active=''
            if old:
                result=subprocess.run(['squeue','-h','-j',old['job_id'],'-o','%T'],text=True,capture_output=True)
                active=result.stdout.strip()
            if active:
                job=old['job_id']
                if previous and old.get('afterok')!=previous:
                    assert set(active.split())=={'PENDING'},(key,active)
                    call('scontrol','update',f'JobId={job}',f'Dependency=afterok:{previous}')
                    old['afterok']=previous
                print(key,job,active.replace('\n',','),'afterok',previous)
            else:
                command=['sbatch','--parsable']
                if previous:command += [f'--dependency=afterok:{previous}']
                command += [str(ROOT/'jobs'/f'{kind}.slurm')]
                if scale:command += [scale]
                job=call(*command).split(';')[0]
                if old:state['history'].append(dict(node=key,**old))
                state['nodes'][key]=dict(job_id=job,afterok=previous,kind=kind,scale=scale)
                print(key,job,'afterok',previous,flush=True)
            tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
            previous=job
        print(json.dumps(state,indent=2))

if __name__=='__main__':main()
