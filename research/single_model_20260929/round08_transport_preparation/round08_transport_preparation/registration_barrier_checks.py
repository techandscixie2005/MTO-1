"""Synthetic stdlib subprocess checks; no scientific child or data access."""
import hashlib,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'REGISTRATION_BARRIER_CHECKS.json';assert not output.exists()
    results={}
    code="from registered_entry import wait_registered; import sys; wait_registered(.5); assert 'torch' not in sys.modules and 'numpy' not in sys.modules; print('RELEASED')"
    for case in ('success','wrong_pid','wrong_binding','closed_pipe','timeout','missing_environment'):
        r,w=os.pipe();binding='a'*64
        env=dict(os.environ,MTO_REGISTRATION_FD=str(r),MTO_REGISTRATION_BINDING=binding)
        if case=='missing_environment':env.pop('MTO_REGISTRATION_FD')
        child=subprocess.Popen([sys.executable,'-c',code],cwd=ROOT,env=env,pass_fds=(r,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        os.close(r)
        if case not in ('closed_pipe','timeout','missing_environment'):
            payload={'registered':True,'pid':child.pid,'binding_sha256':binding}
            if case=='wrong_pid':payload['pid']+=1
            if case=='wrong_binding':payload['binding_sha256']='b'*64
            os.write(w,(json.dumps(payload)+'\n').encode())
        if case!='timeout':os.close(w)
        stdout,stderr=child.communicate(timeout=5)
        if case=='timeout':os.close(w)
        passed=(child.returncode==0 and stdout.strip()=='RELEASED') if case=='success' else (child.returncode!=0 and 'RELEASED' not in stdout)
        assert passed,(case,child.returncode,stdout,stderr)
        results[case]={'passed':True,'returncode':child.returncode,'scientific_stage_called':False}
    hashes={str(ROOT/name):hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('registered_entry.py','registration_barrier_checks.py')}
    result={'passed':True,'checks':results,'source_hashes':hashes,'model_data_imports':0,'optimizer_updates':0,'production_authorized':False}
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
