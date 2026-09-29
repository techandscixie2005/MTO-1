"""Launch only after independent review and archive-first remote publication."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha
from launch import admit,EXPECTED,PYTHON

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
    p.add_argument('--publication-receipt',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['trace','raw_f']);args=p.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text());mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    mh=sha(ROOT/'FROZEN_MANIFEST.json');review=json.loads(args.review.read_text())
    assert review['passed']
    assert review['frozen_manifest_sha256']==mh
    assert review['source_hashes']==mf['source_hashes'],'Review must cover the exact frozen closure'
    for path,expected in review['source_hashes'].items():assert sha(path)==expected
    for path,expected in mf['source_hashes'].items():assert sha(path)==expected
    pub=json.loads(args.publication_receipt.read_text())
    assert pub['download_before_stage_before_commit_push'] is True and pub['remote_verified'] is True
    assert pub['frozen_manifest_sha256']==mh
    lock=(ROOT/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=[]
    for arm in args.arms:
        assert arm in cfg['arms'];gpu=cfg['gpus'][arm]
        rd=ROOT/'runs'/arm;rd.mkdir(parents=True,exist_ok=True)
        assert not (rd/'FIT_COMPLETE.json').exists(),('Already complete',arm)
        glock=open('/tmp/mto_pouter_gpu_%d.lock'%gpu,'w');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission=admit(gpu);attempt='round02_%s_%d'%(arm,time.time_ns())
        (rd/(attempt+'_gpu_admission.xml')).write_text(admission)
        if (rd/'FAILED.json').exists():os.replace(rd/'FAILED.json',rd/(attempt+'_previous_failure.json'))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],PYTHONUNBUFFERED='1',
            PYTHONWARNINGS='ignore::FutureWarning,ignore::UserWarning',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=(rd/'train.log').open('a')
        command=[PYTHON,str(ROOT/'train_frozen.py'),'--arm',arm]
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            pass_fds=(glock.fileno(),),start_new_session=True)
        receipt={'arm':arm,'attempt':attempt,'pid':process.pid,'gpu':gpu,'gpu_uuid':EXPECTED[gpu],
            'manifest_sha256':mh,'review_sha256':sha(args.review),'command':command,
            'publication_receipt':str(args.publication_receipt),'publication_receipt_sha256':sha(args.publication_receipt),
            'launched_at_unix':time.time()}
        (rd/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        relative='round02_frozen/runs/'+arm
        command=['/usr/bin/python3',str(PARENT/'ops/register_run.py'),'--id',attempt,'--pid',str(process.pid),
            '--run-dir',relative,'--gpu-uuid',EXPECTED[gpu]]
        for flag,name in (('status','status.json'),('log','train.log'),('completion','FIT_COMPLETE.json'),('failure','FAILED.json')):
            command+=['--'+flag+'-path',relative+'/'+name]
        for name in ('last.pt','best.pt'):command+=['--checkpoint-path',relative+'/'+name]
        registered=subprocess.run(command,text=True,capture_output=True)
        receipt['monitor_registration_returncode']=registered.returncode
        (rd/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if registered.returncode:raise RuntimeError('Owned PID %d registration failed: %s'%(process.pid,registered.stderr))
        result.append(receipt);glock.close();log.close()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
