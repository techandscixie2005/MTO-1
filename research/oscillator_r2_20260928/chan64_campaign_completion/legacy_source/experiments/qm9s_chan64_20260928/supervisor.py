"""Finite-attempt local queue. Locks and GPU checks prevent duplicate workers."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parent
def save(obj,p):
    p=pathlib.Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)
def health():
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True);root=ET.fromstring(xml);rows={}
    for i,g in enumerate(root.findall('gpu')):
        ecc=[]
        for field in g.findall('ecc_errors/aggregate/*')+g.findall('ecc_errors/volatile/*'):
            if field.text and field.text.strip().isdigit():ecc.append(int(field.text.strip()))
            for leaf in field:
                if leaf.text and leaf.text.strip().isdigit():ecc.append(int(leaf.text.strip()))
        remap=g.find('remapped_rows');r={c.tag:(c.text or '').strip() for c in remap} if remap is not None else {}
        clean=bool(ecc) and sum(ecc)==0 and all(r.get(k)=='No' for k in ('remapped_row_pending','remapped_row_failure'))
        clean=clean and all(r.get(k)=='0' for k in ('remapped_row_corr','remapped_row_unc'))
        clean=clean and all(g.findtext('ecc_errors/'+k)=='No' for k in ('channel_repair_pending','tpc_repair_pending'))
        rows[i]=dict(uuid=g.findtext('uuid'),ecc_counts=ecc,clean=clean,row_remapper=r,memory_MiB=int(g.findtext('fb_memory_usage/used').split()[0]))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    return rows,busy,xml
def alive(pid,script,name=None):
    try:
        cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        cwd=pathlib.Path(f'/proc/{pid}/cwd').resolve()
        return cwd==ROOT and script in cmd and (name is None or name in cmd)
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False
def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(dict(pid=os.getpid(),time=time.time()),ROOT/'supervisor_receipt.json')
    assert json.loads((ROOT/'reports/preflight.json').read_text())['passed']
    assert json.loads((ROOT/'reports/system_preflight.json').read_text())['passed']
    campaign=json.loads((ROOT/'campaign.json').read_text());procs={};handles=[];last_health=None
    while True:
        h,busy,xml=health();last_health=h
        if not (ROOT/'reports/gpu_health_submit.xml').exists():(ROOT/'reports/gpu_health_submit.xml').write_text(xml)
        status={};reserved=set()
        for n in campaign['groups']:
            out=ROOT/'runs'/n;receipt=out/'launch_receipt.json';info=json.loads(receipt.read_text()) if receipt.exists() else {}
            pid=info.get('pid');running=bool(pid and alive(pid,'trainer.py',n));gpu=info.get('gpu')
            if running:
                reserved.add(gpu)
                if not h[gpu]['clean'] and not (out/'GPU_STOP.json').exists():
                    os.kill(pid,signal.SIGTERM);save(dict(time=time.time(),gpu=gpu,reason='GPU health changed; only this worker receives SIGTERM'),out/'GPU_STOP.json')
                event='RUNNING'
            elif (out/'FIT_COMPLETE.json').exists():event='FIT_COMPLETE'
            elif (out/'FAILED.json').exists() or (out/'GPU_STOP.json').exists():event='FAILED'
            elif receipt.exists():
                if not (out/'PROCESS_EXIT.json').exists():save(dict(time=time.time(),pid=pid,complete=False),out/'PROCESS_EXIT.json')
                event='STOPPED_REQUIRES_EXPLICIT_RESUME'
            else:event='QUEUED'
            status[n]=dict(state=event,pid=pid,gpu=gpu,log=str(ROOT/'logs'/f'{n}.log'))
        for n in campaign['groups']:
            if status[n]['state']!='QUEUED':continue
            for gpu in campaign['gpu_pool']:
                if gpu in reserved or not h[gpu]['clean'] or h[gpu]['uuid'] in busy or h[gpu]['memory_MiB']>200:continue
                gpu_lock=open(f'/tmp/mto_pouter_gpu_{gpu}.lock','w')
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:gpu_lock.close();continue
                h2,b2,_=health()
                if not h2[gpu]['clean'] or h2[gpu]['uuid'] in b2 or h2[gpu]['memory_MiB']>200:gpu_lock.close();continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONWARNINGS='ignore',PYTHONDONTWRITEBYTECODE='1')
                log=(ROOT/'logs'/f'{n}.log').open('a');handles.append(log)
                p=subprocess.Popen([sys.executable,'trainer.py',n],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_lock.fileno(),),start_new_session=True)
                gpu_lock.close();procs[n]=p;reserved.add(gpu)
                receipt=dict(name=n,pid=p.pid,gpu=gpu,gpu_uuid=h[gpu]['uuid'],time=time.time(),resume=(ROOT/'runs'/n/'last.pt').exists())
                save(receipt,ROOT/'runs'/n/'launch_receipt.json');status[n].update(state='STARTING',pid=p.pid,gpu=gpu)
                print('LAUNCHED',n,p.pid,gpu,flush=True);break
        for p in procs.values():p.poll()
        save(dict(time=time.time(),pid=os.getpid(),runs=status,health=h),ROOT/'supervisor_status.json')
        if all(v['state']=='FIT_COMPLETE' for v in status.values()):
            if (ROOT/'reports/ANALYSIS_COMPLETE.json').exists() or (ROOT/'reports/ANALYSIS_FAILED.json').exists():return
            for gpu in campaign['gpu_pool']:
                if not h[gpu]['clean'] or h[gpu]['uuid'] in busy or h[gpu]['memory_MiB']>200:continue
                with (ROOT/'logs/evaluate.log').open('a') as log:
                    p=subprocess.Popen([sys.executable,'evaluate.py'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTHONWARNINGS='ignore',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
                    save(dict(pid=p.pid,gpu=gpu,time=time.time()),ROOT/'reports/evaluation_receipt.json');code=p.wait()
                if code:save(dict(exit_code=code,time=time.time()),ROOT/'reports/ANALYSIS_FAILED.json')
                return
        if all(v['state'] in ('FIT_COMPLETE','FAILED','STOPPED_REQUIRES_EXPLICIT_RESUME') for v in status.values()):
            save(dict(time=time.time(),reason='At least one group failed or stopped; evaluation withheld',groups=status),ROOT/'reports/CAMPAIGN_BLOCKED.json');return
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except BlockingIOError:print('Supervisor already active',flush=True)
    except BaseException:
        save(dict(traceback=traceback.format_exc(),time=time.time()),ROOT/'reports/SUPERVISOR_FAILED.json');raise
