"""Owned one-GPU bounded engineering stage, requiring source-review receipt."""
import argparse,fcntl,json,os,subprocess,sys,time
from common import ROOT,CAMPAIGN,sha,read,atomic_json
from resources import EXPECTED,PYTHON,admit
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',required=True,type=int,choices=[1]);args=parser.parse_args()
    os.chdir(ROOT);review_path=ROOT/'TECHNICAL_SOURCE_REVIEW.json';review=read(review_path);rh=sha(review_path)
    assert review['passed'] and review['scope']=='round07_128_train_9_discarded_updates'
    decision_path=ROOT/'ROUND07_PREPARATION_DECISION.md';dh=sha(decision_path)
    assert dh==review['root_preparation_decision_sha256']=='637a3cd3006d3f5b84f8ca5314b6213b0bc3017dffeeafb7ed51c2a4c594c60b'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    assert read(ROOT/'CPU_PREFLIGHT.json')['passed'] and not (ROOT/'GPU_PREFLIGHT.json').exists()
    assert not (ROOT/'private_preflight').exists(),'Inspect prior attempt; never silently repeat updates'
    out=ROOT/'ops/gpu_preflight_attempt';out.mkdir(parents=True,exist_ok=False)
    with open('/tmp/mto_pouter_gpu_%d.lock'%args.gpu,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);xml=admit(args.gpu);(out/'GPU_ADMISSION.xml').write_text(xml)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[args.gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore::UserWarning,ignore::FutureWarning')
        command=[PYTHON,str(ROOT/'gpu_preflight.py')]
        launch={'wrapper_identity':process_identity(os.getpid()),'review_sha256':rh,'command':command,
            'gpu':args.gpu,'gpu_uuid':EXPECTED[args.gpu],'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED')}}
        atomic_json(launch,out/'LAUNCH_RECEIPT.json')
        with (out/'stage.log').open('x') as stream:
            started=time.time();child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,pass_fds=(lock.fileno(),))
            launch.update(child_identity=process_identity(child.pid),child_pid=child.pid,started_unix=started)
            atomic_json(launch,out/'LAUNCH_RECEIPT.json');atomic_json({'state':'running','pid':child.pid},out/'status.json')
            rel=out.relative_to(CAMPAIGN).as_posix()
            register=['/usr/bin/python3',str(CAMPAIGN/'ops/register_run.py'),'--id','round07_technical_preflight',
                '--pid',str(child.pid),'--run-dir',rel,'--gpu-uuid',EXPECTED[args.gpu],
                '--status-path',rel+'/status.json','--log-path',rel+'/stage.log','--completion-path',rel+'/COMPLETE.json','--failure-path',rel+'/FAILED.json']
            registration=subprocess.run(register,capture_output=True,text=True)
            (out/'REGISTRATION_TOOL.log').write_text(registration.stdout+registration.stderr)
            launch['registration_returncode']=registration.returncode;atomic_json(launch,out/'LAUNCH_RECEIPT.json')
            code=child.wait()
        passed=code==0 and registration.returncode==0 and sha(review_path)==rh and sha(decision_path)==dh and (ROOT/'GPU_PREFLIGHT.json').exists()
        terminal={'passed':passed,'exit_code':code,'registration_returncode':registration.returncode,
            'review_unchanged':sha(review_path)==rh,'root_decision_unchanged':sha(decision_path)==dh,'seconds':time.time()-started,'child_identity':launch['child_identity'],
            'log_sha256':sha(out/'stage.log'),'preflight_sha256':sha(ROOT/'GPU_PREFLIGHT.json') if (ROOT/'GPU_PREFLIGHT.json').exists() else None}
        atomic_json(terminal,out/('COMPLETE.json' if passed else 'FAILED.json'));atomic_json(terminal,out/'status.json')
        print(json.dumps(terminal),flush=True)
        if not passed:raise SystemExit(1)
if __name__=='__main__':main()
