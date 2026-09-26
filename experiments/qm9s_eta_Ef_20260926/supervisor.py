"""Only starts/stops this campaign's workers. Never resets or reconfigures GPUs."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
def save(obj,path):
    p=pathlib.Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
def gpu_health():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,memory.used','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in s.strip().splitlines():
        i,uid,v,a,mem=[q.strip() for q in line.split(',')]
        rows[int(i)]=dict(uuid=uid,volatile=int(v),aggregate=int(a),memory_MiB=int(mem))
    return rows
def busy_uuids():
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    return {line.split(',')[0].strip() for line in s.strip().splitlines() if ',' in line}
def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    procs={};attempted=set();files=[];initial=gpu_health();save(initial,ROOT/'reports/gpu_health_initial.json')
    while True:
        campaign=json.loads((ROOT/'campaign.json').read_text());health=gpu_health();busy=busy_uuids()
        for spec in campaign['runs']:
            name=spec['name'];gpu=spec['gpu'];out=ROOT/'runs'/name;out.mkdir(exist_ok=True)
            if name not in attempted and not (out/'FIT_COMPLETE.json').exists() and not (out/'FAILED.json').exists():
                if health[gpu]['volatile'] or health[gpu]['aggregate'] or health[gpu]['uuid'] in busy or health[gpu]['memory_MiB']>200:
                    print('WAIT_GPU',name,gpu,health[gpu],flush=True);continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                log=(ROOT/'logs'/f'{name}.log').open('a');files.append(log)
                p=subprocess.Popen([sys.executable,spec['trainer'],name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                procs[name]=(p,gpu);attempted.add(name)
                save(dict(name=name,gpu=gpu,pid=p.pid,time=time.time(),trainer=spec['trainer']),out/'launch_receipt.json')
                print('LAUNCHED',name,p.pid,gpu,flush=True)
        status={}
        for name,(p,gpu) in procs.items():
            code=p.poll();out=ROOT/'runs'/name
            if code is None and (health[gpu]['volatile']>initial[gpu]['volatile'] or health[gpu]['aggregate']>initial[gpu]['aggregate']):
                p.send_signal(signal.SIGTERM)
                save(dict(reason='New uncorrectable ECC; SIGTERM sent to this run only',gpu=gpu,health=health[gpu],time=time.time()),out/'GPU_STOP.json')
                print('GPU_STOP',name,flush=True)
            if code is not None and not (out/'FIT_COMPLETE.json').exists() and not (out/'FAILED.json').exists() and not (out/'PROCESS_EXIT.json').exists():
                save(dict(exit_code=code,complete=False,time=time.time()),out/'PROCESS_EXIT.json')
            status[name]=dict(pid=p.pid,gpu=gpu,exit_code=code,complete=(out/'FIT_COMPLETE.json').exists())
        save(dict(time=time.time(),runs=status,health=health,detanet_configuration_status=campaign['detanet_configuration_status']),ROOT/'supervisor_status.json')
        if campaign['detanet_configuration_status']=='confirmed' and len(campaign['runs'])==4 and all((ROOT/'runs'/s['name']/'FIT_COMPLETE.json').exists() for s in campaign['runs']):
            if not (ROOT/'reports/ANALYSIS_COMPLETE.json').exists():
                # All training workers exited; evaluation uses the first currently idle healthy assigned card.
                avail=[i for i in (1,2,4,5,6,0) if not health[i]['aggregate'] and health[i]['uuid'] not in busy]
                if avail:
                    with (ROOT/'logs/evaluate.log').open('a') as f:
                        code=subprocess.call([sys.executable,'evaluate.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(avail[0]),OMP_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
                    if code:save(dict(exit_code=code,time=time.time()),ROOT/'reports/ANALYSIS_FAILED.json')
                    else:print('CAMPAIGN_COMPLETE',flush=True)
                    return
            else:return
        time.sleep(30)
if __name__=='__main__':
    try:main()
    except BaseException:
        save(dict(traceback=traceback.format_exc(),time=time.time()),ROOT/'reports/SUPERVISOR_FAILED.json');raise
