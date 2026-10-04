"""Owned one-GPU bounded engineering stage, requiring source-review receipt."""
import argparse,fcntl,json,os,subprocess,sys,time
from unittest.mock import patch
import resources
from common import ROOT,CAMPAIGN,sha,read,atomic_json
from resources import EXPECTED,PYTHON,admit
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity

ORDER=(1,2,4,6)

def select_locked_device(out):
    """One bounded inventory/admission pass. The selected lock stays held."""
    observations=[]
    for gpu in ORDER:
        lock=open('/tmp/mto_pouter_gpu_%d.lock'%gpu,'w')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            original_probe=subprocess.check_output
            def capture(command,**kwargs):
                assert command==['nvidia-smi','-q','-x','-i',str(gpu)]
                xml=original_probe(command,**kwargs)
                (out/('GPU%d_PROBE.xml'%gpu)).write_text(xml)
                return xml
            # Capture the exact single probe before unchanged admit assertions.
            with patch.object(resources.subprocess,'check_output',capture):xml=admit(gpu)
        except (BlockingIOError,AssertionError) as exc:
            observations.append({'gpu':gpu,'eligible':False,'reason':repr(exc),'observed_unix':time.time(),
                'probe_path':('GPU%d_PROBE.xml'%gpu) if (out/('GPU%d_PROBE.xml'%gpu)).exists() else None,
                'probe_sha256':sha(out/('GPU%d_PROBE.xml'%gpu)) if (out/('GPU%d_PROBE.xml'%gpu)).exists() else None})
            lock.close();atomic_json({'ordered_allowed':list(ORDER),'observations':observations},out/'RESOURCE_SELECTION.json')
            continue
        except BaseException:
            lock.close();raise
        observations.append({'gpu':gpu,'eligible':True,'uuid':EXPECTED[gpu],'observed_unix':time.time(),
            'probe_path':'GPU%d_PROBE.xml'%gpu,'probe_sha256':sha(out/('GPU%d_PROBE.xml'%gpu))})
        atomic_json({'ordered_allowed':list(ORDER),'observations':observations,'selected_gpu':gpu,
            'selected_uuid':EXPECTED[gpu],'selection_under_exclusive_lock':True},out/'RESOURCE_SELECTION.json')
        return lock,gpu,xml
    return None,None,None

def main():
    parser=argparse.ArgumentParser();parser.parse_args()
    os.chdir(ROOT);review_path=ROOT/'TECHNICAL_SOURCE_REVIEW.json';review=read(review_path);rh=sha(review_path)
    assert review['passed'] and review['scope']=='round10_128_train_6_discarded_updates'
    decision_path=ROOT/'ROUND10_PREPARATION_DECISION.md';dh=sha(decision_path)
    assert dh==review['root_preparation_decision_sha256']=='4d23e6ab826dd2f4df1f4e72939529ff823fd2210a897a0ea4ba0f5dd6494890'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    assert read(ROOT/'CPU_PREFLIGHT.json')['passed'] and not (ROOT/'GPU_PREFLIGHT.json').exists()
    assert read(ROOT/'CPU_PREFLIGHT.json')['parameter_roster_sha256']==sha(ROOT/'PARAMETER_ROSTER.json')
    assert read(ROOT/'PARAMETER_ROSTER.json')['frozen_before_any_optimizer_update']
    assert not (ROOT/'private_preflight').exists(),'Inspect prior attempt; never silently repeat updates'
    out=ROOT/'ops/gpu_preflight_attempt';out.mkdir(parents=True,exist_ok=False)
    lock,gpu,xml=select_locked_device(out)
    if lock is None:
        atomic_json({'held':True,'scientific_stage_started':False,'optimizer_updates':0,
            'reason':'No eligible device in one ordered admission pass; no automatic retry'},out/'HOLD.json')
        raise SystemExit(2)
    with lock:
        (out/'GPU_ADMISSION.xml').write_text(xml)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore::UserWarning,ignore::FutureWarning')
        barrier_read,barrier_write=os.pipe()
        env.update(MTO_REGISTRATION_FD=str(barrier_read),MTO_REGISTRATION_BINDING=rh)
        command=[PYTHON,str(ROOT/'registered_entry.py'),str(ROOT/'gpu_preflight.py')]
        launch={'wrapper_identity':process_identity(os.getpid()),'review_sha256':rh,'command':command,
            'gpu':gpu,'gpu_uuid':EXPECTED[gpu],'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED')}}
        atomic_json(launch,out/'LAUNCH_RECEIPT.json')
        with (out/'stage.log').open('x') as stream:
            started=time.time();child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,pass_fds=(lock.fileno(),barrier_read))
            os.close(barrier_read)
            launch.update(child_identity=process_identity(child.pid),child_pid=child.pid,started_unix=started)
            atomic_json(launch,out/'LAUNCH_RECEIPT.json');atomic_json({'state':'running','pid':child.pid},out/'status.json')
            rel=out.relative_to(CAMPAIGN).as_posix()
            register=['/usr/bin/python3',str(CAMPAIGN/'ops/register_run.py'),'--id','round10_technical_preflight',
                '--pid',str(child.pid),'--run-dir',rel,'--gpu-uuid',EXPECTED[gpu],
                '--status-path',rel+'/status.json','--log-path',rel+'/stage.log','--completion-path',rel+'/COMPLETE.json','--failure-path',rel+'/FAILED.json']
            try:registration=subprocess.run(register,capture_output=True,text=True,timeout=45)
            except subprocess.TimeoutExpired:
                registration=subprocess.CompletedProcess(register,124,'','Registration timed out; child barrier remains closed.\n')
            (out/'REGISTRATION_TOOL.log').write_text(registration.stdout+registration.stderr)
            launch['registration_returncode']=registration.returncode;atomic_json(launch,out/'LAUNCH_RECEIPT.json')
            try:
                if registration.returncode==0:
                    os.write(barrier_write,(json.dumps({'registered':True,'pid':child.pid,'binding_sha256':rh})+'\n').encode())
                    launch['registration_barrier_released']=True
                else:launch['registration_barrier_released']=False
            finally:os.close(barrier_write)
            atomic_json(launch,out/'LAUNCH_RECEIPT.json')
            if registration.returncode==0:code=child.wait()
            else:
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate();code=child.wait(timeout=10)
        passed=code==0 and registration.returncode==0 and sha(review_path)==rh and sha(decision_path)==dh and (ROOT/'GPU_PREFLIGHT.json').exists()
        terminal={'passed':passed,'exit_code':code,'registration_returncode':registration.returncode,
            'review_unchanged':sha(review_path)==rh,'root_decision_unchanged':sha(decision_path)==dh,'seconds':time.time()-started,'child_identity':launch['child_identity'],
            'log_sha256':sha(out/'stage.log'),'preflight_sha256':sha(ROOT/'GPU_PREFLIGHT.json') if (ROOT/'GPU_PREFLIGHT.json').exists() else None}
        atomic_json(terminal,out/('COMPLETE.json' if passed else 'FAILED.json'));atomic_json(terminal,out/'status.json')
        print(json.dumps(terminal),flush=True)
        if not passed:raise SystemExit(1)
if __name__=='__main__':main()
