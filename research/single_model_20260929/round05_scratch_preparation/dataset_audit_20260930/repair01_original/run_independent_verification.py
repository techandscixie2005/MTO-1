"""Owned CPU wrapper for the one independent target-free saved-split check."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CAMPAIGN=ROOT.parent
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import atomic_json,process_identity
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert {k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')}=={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    os.chdir(ROOT);output=ROOT/'INDEPENDENT_SPLIT_VERIFICATION.json'
    assert not output.exists(),'Independent saved-split check already complete'
    assert (ROOT/'SPLIT_MANIFEST.json').exists()
    auth=ROOT/'MATERIALIZATION_REVIEW.json';digest=sha(auth)
    assert json.loads(auth.read_text())['passed']
    run=ROOT/'ops/independent_verification_attempt';run.mkdir(exist_ok=False)
    with (run/'owner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[PYTHON,str(ROOT/'independent_verify_partition.py')]
        receipt={'wrapper_identity':process_identity(os.getpid()),'wrapper_sha256':sha(__file__),
            'verifier_sha256':sha(ROOT/'independent_verify_partition.py'),'materialization_review_sha256':digest,
            'split_manifest_sha256':sha(ROOT/'SPLIT_MANIFEST.json'),'command':command,'cwd':str(ROOT),
            'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
        atomic_json(run/'LAUNCH_RECEIPT.json',receipt)
        relative=run.relative_to(CAMPAIGN).as_posix()
        cmd=['/usr/bin/python3',str(CAMPAIGN/'ops/register_cpu_stage.py'),'--id','v2_split_independent_verification',
            '--pid',str(os.getpid()),'--run-dir',relative,'--status-path',relative+'/status.json',
            '--log-path',relative+'/stage.log','--completion-path',relative+'/COMPLETE.json','--failure-path',relative+'/FAILED.json',
            '--authorization',auth.relative_to(CAMPAIGN).as_posix(),'--authorization-sha256',digest]
        registered=subprocess.run(cmd,capture_output=True,text=True)
        (run/'REGISTRATION_TOOL.log').write_text(registered.stdout+registered.stderr)
        if registered.returncode:
            atomic_json(run/'FAILED.json',{'stage':'registration','exit_code':registered.returncode});raise SystemExit(registered.returncode)
        with (run/'stage.log').open('x') as stream:
            start=time.time();p=subprocess.Popen(command,env=dict(os.environ),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            receipt.update(child_pid=p.pid,child_identity=process_identity(p.pid),started_at_unix=start)
            atomic_json(run/'LAUNCH_RECEIPT.json',receipt);atomic_json(run/'status.json',{'state':'running','child_pid':p.pid})
            code=p.wait()
        terminal={'exit_code':code,'duration_seconds':time.time()-start,'child_identity':receipt['child_identity'],
            'materialization_review_unchanged':sha(auth)==digest,'split_manifest_unchanged':sha(ROOT/'SPLIT_MANIFEST.json')==receipt['split_manifest_sha256'],
            'log_sha256':sha(run/'stage.log'),'output_sha256':sha(output) if output.exists() else None}
        passed=code==0 and output.exists() and terminal['materialization_review_unchanged'] and terminal['split_manifest_unchanged']
        atomic_json(run/('COMPLETE.json' if passed else 'FAILED.json'),terminal)
        atomic_json(run/'status.json',{'state':'complete' if passed else 'failed',**terminal})
        print(json.dumps(terminal),flush=True)
        if not passed:raise SystemExit(1)
if __name__=='__main__':main()
