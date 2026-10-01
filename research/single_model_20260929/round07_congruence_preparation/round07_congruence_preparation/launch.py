"""Future production launcher; exact authority/publication gate before mutation."""
import argparse,fcntl,json,os,subprocess,sys,time
from common import ROOT,CAMPAIGN,read,sha,atomic_json
from execution_gate import verify
from resources import EXPECTED,PYTHON,admit
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity
ASSIGNMENT={'original':1,'scalar':2,'tensor':4}

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True)
    p.add_argument('--arms',nargs='+',choices=list(ASSIGNMENT),default=list(ASSIGNMENT));args=p.parse_args()
    assert len(args.arms)==len(set(args.arms));permit=verify(args.authorization)
    with (ROOT/'launch.lock').open('w') as launchlock:
        fcntl.flock(launchlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in args.arms:
            rd=ROOT/'runs'/arm
            assert not (rd/'FIT_COMPLETE.json').exists(),'Completed arm cannot relaunch'
            # A live same-arm worker holds this separate lock; no PID guessing.
            rd.mkdir(parents=True,exist_ok=True)
            with (rd/'worker.lock').open('w') as check:
                fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu=ASSIGNMENT[arm];glock=open('/tmp/mto_pouter_gpu_%d.lock'%gpu,'w')
            fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);xml=admit(gpu)
            attempt=f'round07_{arm}_{time.time_ns()}';out=rd/'attempts'/attempt;out.mkdir(parents=True,exist_ok=False)
            (out/'GPU_ADMISSION.xml').write_text(xml)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore::UserWarning,ignore::FutureWarning')
            cmd=[PYTHON,str(ROOT/'train.py'),'--arm',arm,'--authorization',str(args.authorization)]
            with (out/'train.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(glock.fileno(),),start_new_session=True)
            receipt={'attempt':attempt,'arm':arm,'gpu':gpu,'gpu_uuid':EXPECTED[gpu],
                'identity':process_identity(child.pid),'command':cmd,'manifest_sha256':permit['manifest_sha256'],
                'authorization_sha256':permit['authorization_sha256'],'review_sha256':permit['review_sha256'],
                'publication_sha256':permit['publication_sha256'],'started_unix':time.time(),
                'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED')}}
            atomic_json(receipt,out/'LAUNCH_RECEIPT.json')
            rel=rd.relative_to(CAMPAIGN).as_posix();logrel=(out/'train.log').relative_to(CAMPAIGN).as_posix()
            registration=['/usr/bin/python3',str(CAMPAIGN/'ops/register_run.py'),'--id',attempt,'--pid',str(child.pid),
                '--run-dir',rel,'--gpu-uuid',EXPECTED[gpu],'--status-path',rel+'/status.json','--log-path',logrel,
                '--completion-path',rel+'/FIT_COMPLETE.json','--failure-path',rel+'/FAILED.json',
                '--checkpoint-path',rel+'/last.pt','--checkpoint-path',rel+'/best.pt']
            result=subprocess.run(registration,capture_output=True,text=True)
            (out/'REGISTRATION_TOOL.log').write_text(result.stdout+result.stderr)
            receipt['registration_returncode']=result.returncode;atomic_json(receipt,out/'LAUNCH_RECEIPT.json')
            glock.close();print(json.dumps(receipt),flush=True)
            if result.returncode:raise RuntimeError('Owned child remains running; registration repair required, no automatic signal')
if __name__=='__main__':main()
