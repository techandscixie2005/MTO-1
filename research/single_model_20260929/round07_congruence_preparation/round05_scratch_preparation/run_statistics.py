"""Owned, CPU-only reviewed TRAIN-statistics stage; no automatic retry."""
import fcntl,os,subprocess,sys,time,json
from pathlib import Path
from common import ROOT,CAMPAIGN,sha,read,atomic_json,require_cpu
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'

def main():
    require_cpu();os.chdir(ROOT)
    auth=ROOT/'TRAIN_READER_REVIEW.json';h=sha(auth);review=read(auth)
    assert review['passed'] and review['phase']=='train_statistics_preparation'
    for n,value in review['source_hashes'].items():assert sha(ROOT/n)==value
    assert not (ROOT/'TRAIN_STATISTICS.json').exists()
    out=ROOT/'ops/statistics_attempt';out.mkdir(parents=True,exist_ok=False)
    with (out/'owner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cmd=[PYTHON,str(ROOT/'prepare_train_statistics.py')]
        receipt={'wrapper_identity':process_identity(os.getpid()),'command':cmd,'review_sha256':h,
            'wrapper_sha256':sha(__file__),'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
        atomic_json(receipt,out/'LAUNCH_RECEIPT.json')
        rel=out.relative_to(CAMPAIGN).as_posix()
        registration=['/usr/bin/python3',str(CAMPAIGN/'ops/register_cpu_stage.py'),'--id','round05_train_statistics',
            '--pid',str(os.getpid()),'--run-dir',rel,'--status-path',rel+'/status.json','--log-path',rel+'/stage.log',
            '--completion-path',rel+'/COMPLETE.json','--failure-path',rel+'/FAILED.json',
            '--authorization',auth.relative_to(CAMPAIGN).as_posix(),'--authorization-sha256',h]
        result=subprocess.run(registration,capture_output=True,text=True)
        (out/'REGISTRATION_TOOL.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            atomic_json({'stage':'registration','exit_code':result.returncode},out/'FAILED.json');raise SystemExit(result.returncode)
        with (out/'stage.log').open('x') as stream:
            start=time.time();child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            receipt.update(child_identity=process_identity(child.pid),child_pid=child.pid,started_unix=start)
            atomic_json(receipt,out/'LAUNCH_RECEIPT.json');atomic_json({'state':'running','child_pid':child.pid},out/'status.json')
            code=child.wait()
        complete=code==0 and sha(auth)==h and (ROOT/'TRAIN_STATISTICS_AUDIT.json').exists()
        terminal={'passed':complete,'exit_code':code,'review_unchanged':sha(auth)==h,'seconds':time.time()-start,
            'child_identity':receipt['child_identity'],'log_sha256':sha(out/'stage.log'),
            'outputs':{n:sha(ROOT/n) for n in ('TRAIN_STATISTICS.json','TRAIN_STATISTICS_AUDIT.json') if (ROOT/n).exists()}}
        atomic_json(terminal,out/('COMPLETE.json' if complete else 'FAILED.json'));atomic_json(terminal,out/'status.json')
        print(json.dumps(terminal),flush=True)
        if not complete:raise SystemExit(1)
if __name__=='__main__':main()
