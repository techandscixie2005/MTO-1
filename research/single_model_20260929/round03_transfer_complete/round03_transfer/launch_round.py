"""Launch the one clean source only after exact review and archive-first push."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from runtime import ROOT,sha,verify_manifest,atomic_json
sys.path.insert(0,str(ROOT.parent))
from launch import admit,EXPECTED,PYTHON

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review',required=True);parser.add_argument('--publication-receipt',required=True)
    args=parser.parse_args();manifest,mh=verify_manifest();cfg=manifest['config']
    review=json.load(open(args.review));assert review['passed'] and review['frozen_manifest_sha256']==mh
    assert review['source_hashes']==manifest['source_hashes']
    publication=json.load(open(args.publication_receipt))
    assert publication['download_before_stage_before_commit_push'] is True and publication['remote_verified'] is True
    assert publication['frozen_manifest_sha256']==mh
    assert publication['independent_review_sha256']==sha(args.review)
    previous=json.loads((ROOT.parent/'ops/ROUND02_COMPLETION_PUBLICATION_RECEIPT.json').read_text())
    assert previous['remote_verified'] is True
    lock=open(ROOT/'launch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=cfg['gpu'];glock=open('/tmp/mto_pouter_gpu_%d.lock'%gpu,'w');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=admit(gpu);rd=ROOT/'runs/source33';rd.mkdir(parents=True,exist_ok=True)
    assert not (rd/'FIT_COMPLETE.json').exists(),'Source already complete'
    attempt='round03_source33_%d'%time.time_ns();(rd/(attempt+'_gpu_admission.xml')).write_text(admission)
    if (rd/'FAILED.json').exists():os.replace(rd/'FAILED.json',rd/(attempt+'_previous_failure.json'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],PYTHONUNBUFFERED='1',
        PYTHONWARNINGS='ignore::FutureWarning,ignore::UserWarning',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    log=open(rd/'train.log','a');command=[PYTHON,str(ROOT/'train_source.py')]
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
        pass_fds=(glock.fileno(),),start_new_session=True)
    receipt={'attempt':attempt,'pid':process.pid,'gpu':gpu,'gpu_uuid':EXPECTED[gpu],
        'manifest_sha256':mh,'review_sha256':sha(args.review),'publication_receipt_sha256':sha(args.publication_receipt),
        'command':command,'launched_at_unix':time.time(),'fixed_source_epochs':33}
    atomic_json(receipt,rd/'LAUNCH_RECEIPT.json')
    relative='round03_transfer/runs/source33'
    command=['/usr/bin/python3',str(ROOT.parent/'ops/register_run.py'),'--id',attempt,'--pid',str(process.pid),
        '--run-dir',relative,'--gpu-uuid',EXPECTED[gpu]]
    for flag,name in (('status','status.json'),('log','train.log'),('completion','FIT_COMPLETE.json'),('failure','FAILED.json')):
        command+=['--'+flag+'-path',relative+'/'+name]
    for name in ('last.pt','source_final.pt'):command+=['--checkpoint-path',relative+'/'+name]
    result=subprocess.run(command,capture_output=True,text=True);receipt['monitor_registration_returncode']=result.returncode
    atomic_json(receipt,rd/'LAUNCH_RECEIPT.json')
    if result.returncode:raise RuntimeError('Owned PID %d registration failed: %s'%(process.pid,result.stderr))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
