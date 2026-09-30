"""Persistent owned CPU wrapper for reviewed identity audit/materialization; no auto retry."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import atomic_json,process_identity
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('audit','materialize'));p.add_argument('--attempt-id',default='');args=p.parse_args()
    assert not args.attempt_id or re.fullmatch('[a-z0-9_]+',args.attempt_id)
    suffix=('_'+args.attempt_id) if args.attempt_id else ''
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
    output=ROOT/('CORPUS_AUDIT.json' if args.mode=='audit' else 'SPLIT_MANIFEST.json')
    if output.exists():raise RuntimeError('Output already exists; inspect/reuse it, do not duplicate this stage')
    ops=ROOT/'ops';ops.mkdir(exist_ok=True)
    with (ops/'builder.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=ops/(args.mode+'_attempt'+suffix);run.mkdir(exist_ok=False)
        auth=ROOT/('PRE_CORPUS_REVIEW.json' if args.mode=='audit' else 'MATERIALIZATION_REVIEW.json')
        review=json.loads(auth.read_text());assert review['passed'] and review['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
        digest=sha(auth);relative=run.relative_to(CAMPAIGN).as_posix()
        command=[PYTHON,str(ROOT/'build_partition.py'),args.mode]
        receipt={'mode':args.mode,'command':command,'cwd':str(ROOT),'wrapper_identity':process_identity(os.getpid()),'review_sha256':digest,'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'wrapper_sha256':sha(__file__),'created_at_unix':time.time(),'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
        atomic_json(run/'LAUNCH_RECEIPT.json',receipt)
        register=['/usr/bin/python3',str(CAMPAIGN/'ops/register_cpu_stage.py'),'--id','v2_split_'+args.mode+suffix,'--pid',str(os.getpid()),'--run-dir',relative,
            '--status-path',relative+'/status.json','--log-path',relative+'/stage.log','--completion-path',relative+'/COMPLETE.json','--failure-path',relative+'/FAILED.json',
            '--authorization',auth.relative_to(CAMPAIGN).as_posix(),'--authorization-sha256',digest]
        registration=subprocess.run(register,check=True,capture_output=True,text=True)
        (run/'REGISTRATION_TOOL.log').write_text(registration.stdout)
        with (run/'stage.log').open('x') as log:
            start=time.time();child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            receipt.update(child_pid=child.pid,child_identity=process_identity(child.pid),child_started_at_unix=start)
            atomic_json(run/'LAUNCH_RECEIPT.json',receipt);atomic_json(run/'status.json',{'status':'running','pid':child.pid,'identity':receipt['child_identity'],'started_at_unix':start})
            code=child.wait();end=time.time()
        complete={'mode':args.mode,'exit_code':code,'started_at_unix':start,'ended_at_unix':end,'duration_seconds':end-start,'child_identity':receipt['child_identity'],'review_unchanged':sha(auth)==digest,'source_manifest_unchanged':sha(ROOT/'SOURCE_MANIFEST.json')==receipt['source_manifest_sha256'],'log_sha256':sha(run/'stage.log'),'output_sha256':sha(output) if output.exists() else None}
        passed=code==0 and output.exists() and complete['review_unchanged'] and complete['source_manifest_unchanged']
        atomic_json(run/('COMPLETE.json' if passed else 'FAILED.json'),complete)
        atomic_json(run/'status.json',{'status':'complete' if passed else 'failed',**complete})
        print(json.dumps(complete),flush=True)
        if not passed:raise SystemExit(1)
if __name__=='__main__':main()
