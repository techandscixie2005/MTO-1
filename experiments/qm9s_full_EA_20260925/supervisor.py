"""Own only this campaign's processes. Never reset GPUs or touch unrelated jobs."""
import json,os,pathlib,signal,subprocess,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
PY=str(ROOT/'env/bin/python')
def save(obj,path):
    p=ROOT/path;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
def health():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,ecc.errors.uncorrected.volatile.dram','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in raw.splitlines())}
def main():
    assert json.loads((ROOT/'reports/preflight.json').read_text())['status']=='PASS'
    campaign=json.loads((ROOT/'campaign.json').read_text());initial=health();procs={};handles=[]
    for name in campaign['runs']:
        cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text());gpu=cfg['gpu'];assert gpu not in campaign['gpu_excluded'] and initial[gpu]==0
        if (ROOT/'runs'/name/'FIT_COMPLETE.json').exists():continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        f=(ROOT/'logs'/f'{name}.log').open('a');handles.append(f)
        p=subprocess.Popen([PY,'train_ea.py',name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        procs[name]=(p,gpu);print('LAUNCHED',name,gpu,p.pid,flush=True)
    save(dict(started=time.time(),pids={name:dict(pid=p.pid,gpu=gpu) for name,(p,gpu) in procs.items()},initial_ecc=initial),'launch_receipt.json')
    while any(p.poll() is None for p,g in procs.values()):
        try:
            ecc=health()
            for name,(p,g) in procs.items():
                if p.poll() is None and ecc[g]>initial[g]:
                    p.send_signal(signal.SIGTERM);save(dict(name=name,gpu=g,uncorrectable_ecc=ecc[g],time=time.time()),'reports/HARDWARE_ALERT.json')
        except subprocess.SubprocessError:pass
        save(dict(time=time.time(),runs={name:dict(pid=p.pid,gpu=g,returncode=p.poll(),fit_complete=(ROOT/'runs'/name/'FIT_COMPLETE.json').exists()) for name,(p,g) in procs.items()}),'supervisor_status.json')
        time.sleep(60)
    for f in handles:f.close()
    if not all((ROOT/'runs'/name/'FIT_COMPLETE.json').exists() for name in campaign['runs']):
        save(dict(status='ATTENTION_REQUIRED',time=time.time(),incomplete=[name for name in campaign['runs'] if not (ROOT/'runs'/name/'FIT_COMPLETE.json').exists()]),'reports/CAMPAIGN_INCOMPLETE.json');return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'logs/analysis.log').open('a') as f:subprocess.run([PY,'analyze.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        save(dict(error=traceback.format_exc(),time=time.time()),'reports/SUPERVISOR_ERROR.json');raise
