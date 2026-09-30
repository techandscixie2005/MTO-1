"""Operational wrapper: register this CPU owner, run the unchanged reviewed child, record exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROUND=Path(__file__).resolve().parents[1]
SCOPE=ROUND.parent
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'
sys.path.insert(0,str(ROUND))
from common import require_cpu,sha,atomic_json
from execution_gate import verify_execution
sys.path.insert(0,str(SCOPE/'ops'))
from monitor import process_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('fit_export','evaluate'));args=parser.parse_args()
    require_cpu();os.chdir(ROUND)
    authorization=ROUND/'PRODUCTION_EXECUTION_AUTHORIZATION.json'
    manifest=ROUND/'FROZEN_MANIFEST.json';review=ROUND/'INDEPENDENT_PREPARATION_REVIEW.json'
    publication=SCOPE/'ops/ROUND04_PREPARATION_PUBLICATION_RECEIPT.json'
    permit=verify_execution(authorization,manifest,review,publication)
    if args.stage=='evaluate' and (ROUND/'production/VALIDATION_COMPLETE.json').exists():
        raise RuntimeError('The fixed validation comparison is already complete')
    if args.stage=='fit_export' and (ROUND/'production/POSTFIT_GEOMETRY_REPLAY.json').exists():
        raise RuntimeError('Fit/export and post-fit replay are already complete')
    attempt='round04_%s_%d'%(args.stage,time.time_ns())
    relative='round04_fonly_preparation/ops/execution/'+attempt
    directory=SCOPE/relative;directory.mkdir(parents=True)
    command=[PYTHON,str(ROUND/'production_entry.py'),args.stage,
        '--authorization',str(authorization),'--manifest',str(manifest),'--review',str(review),'--publication',str(publication)]
    receipt={'attempt':attempt,'stage':args.stage,'permit':permit,'command':command,'cwd':str(ROUND),
        'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'wrapper_identity':process_identity(os.getpid()),'created_at_unix':time.time(),
        'wrapper_sha256':sha(__file__),'registrar_sha256':sha(SCOPE/'ops/register_cpu_stage.py')}
    atomic_json(receipt,directory/'LAUNCH_RECEIPT.json')
    atomic_json({'state':'registering','stage':args.stage},directory/'status.json')
    (directory/'stage.log').touch()
    registration=['/usr/bin/python3',str(SCOPE/'ops/register_cpu_stage.py'),
        '--id',attempt,'--pid',str(os.getpid()),'--run-dir',relative,
        '--status-path',relative+'/status.json','--log-path',relative+'/stage.log',
        '--completion-path',relative+'/COMPLETE.json','--failure-path',relative+'/FAILED.json',
        '--authorization','round04_fonly_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json',
        '--authorization-sha256',sha(authorization)]
    registered=subprocess.run(registration,capture_output=True,text=True)
    (directory/'REGISTRATION_TOOL.log').write_text(registered.stdout+registered.stderr)
    if registered.returncode:
        atomic_json({'failed':True,'reason':'monitor_registration','returncode':registered.returncode},directory/'FAILED.json')
        raise RuntimeError('CPU registration failed; no scientific child started')
    started=time.time()
    with open(directory/'stage.log','ab') as stream:
        process=subprocess.Popen(command,cwd=ROUND,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT)
        receipt['child_pid']=process.pid;receipt['child_identity']=process_identity(process.pid)
        receipt['child_started_at_unix']=started
        atomic_json(receipt,directory/'LAUNCH_RECEIPT.json')
        atomic_json({'state':'running','stage':args.stage,'child_pid':process.pid,'started_at_unix':started},directory/'status.json')
        code=process.wait()
    ended=time.time();terminal={'attempt':attempt,'stage':args.stage,'permit':permit,'exit_code':code,
        'started_at_unix':started,'ended_at_unix':ended,'duration_seconds':ended-started,
        'child_identity':receipt['child_identity'],'child_pid':process.pid,'log_sha256':sha(directory/'stage.log'),
        'authorization_unchanged':sha(authorization)==permit['authorization_sha256']}
    if code==0:
        names=['COEFFICIENTS_FROZEN.json','EXPORT_COMPLETE.json','POSTFIT_GEOMETRY_REPLAY.json'] if args.stage=='fit_export' else ['VALIDATION_COMPLETE.json']
        terminal['output_receipt_hashes']={name:sha(ROUND/'production'/name) for name in names}
        atomic_json(terminal,directory/'COMPLETE.json')
    else:atomic_json(terminal,directory/'FAILED.json')
    atomic_json({'state':'complete' if code==0 else 'failed','stage':args.stage,'exit_code':code,'duration_seconds':ended-started},directory/'status.json')
    print(json.dumps({'attempt':attempt,'stage':args.stage,'exit_code':code,'duration_seconds':ended-started,
        'receipt':str(directory/('COMPLETE.json' if code==0 else 'FAILED.json'))},indent=2))
    if code:raise SystemExit(code)


if __name__=='__main__':main()
