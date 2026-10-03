"""Future production launcher; exact authority/publication gate before mutation."""
import argparse,fcntl,json,os,subprocess,sys,time
from common import ROOT,CAMPAIGN,read,sha,atomic_json
from execution_gate import verify
from resources import EXPECTED,PYTHON,admit
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity
ASSIGNMENT={'zero_decay':1,'coupled_l2':2}

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
            attempt=f'round09_{arm}_{time.time_ns()}';out=rd/'attempts'/attempt;out.mkdir(parents=True,exist_ok=False)
            (out/'GPU_ADMISSION.xml').write_text(xml)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore::UserWarning,ignore::FutureWarning')
            barrier_read,barrier_write=os.pipe()
            env.update(MTO_REGISTRATION_FD=str(barrier_read),MTO_REGISTRATION_BINDING=permit['authorization_sha256'])
            cmd=[PYTHON,str(ROOT/'registered_entry.py'),str(ROOT/'train.py'),'--arm',arm,'--authorization',str(args.authorization)]
            with (out/'train.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(glock.fileno(),barrier_read),start_new_session=True)
            os.close(barrier_read)
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
            try:result=subprocess.run(registration,capture_output=True,text=True,timeout=45)
            except subprocess.TimeoutExpired:
                result=subprocess.CompletedProcess(registration,124,'','Registration timed out; child barrier remains closed.\n')
            (out/'REGISTRATION_TOOL.log').write_text(result.stdout+result.stderr)
            receipt['registration_returncode']=result.returncode
            try:
                if result.returncode==0:
                    os.write(barrier_write,(json.dumps({'registered':True,'pid':child.pid,'binding_sha256':permit['authorization_sha256']})+'\n').encode())
                    receipt['registration_barrier_released']=True
                else:receipt['registration_barrier_released']=False
            finally:os.close(barrier_write)
            if result.returncode:
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate();code=child.wait(timeout=10)
                receipt['owned_child_exit_code']=code
                atomic_json({'stage':'registration','failed':True,'attempt':attempt,'registration_returncode':result.returncode,
                             'child_exit_code':code,'scientific_stage_released':False},rd/'FAILED.json')
            atomic_json(receipt,out/'LAUNCH_RECEIPT.json')
            glock.close();print(json.dumps(receipt),flush=True)
            if result.returncode:raise RuntimeError('Registration failed; owned child stopped before scientific imports. No automatic retry.')
if __name__=='__main__':main()
