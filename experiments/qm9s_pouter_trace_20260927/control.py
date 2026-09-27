"""Explicit status/submission/resumption; never retries automatically."""
import argparse,json,os,subprocess,sys,time
from supervisor import ROOT,alive,save
def start():
    receipt=ROOT/'supervisor_receipt.json'
    if receipt.exists() and alive(json.loads(receipt.read_text())['pid'],'supervisor.py'):
        print('Supervisor is already active');return
    with (ROOT/'logs/supervisor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'supervisor.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    print('Supervisor submitted PID',p.pid)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','status','resume']);p.add_argument('group',nargs='?',choices=['G1','G2','G3','G4']);a=p.parse_args()
    if a.action=='status':
        for file in ['supervisor_status.json','reports/ANALYSIS_COMPLETE.json','reports/ANALYSIS_FAILED.json']:
            path=ROOT/file
            if path.exists():print(file,path.read_text())
        return
    if a.action=='resume':
        assert a.group,'Name the stopped group explicitly'
        out=ROOT/'runs'/a.group;receipt=out/'launch_receipt.json'
        if receipt.exists():assert not alive(json.loads(receipt.read_text())['pid'],'trainer.py',a.group),'Worker still active'
        assert not (out/'FIT_COMPLETE.json').exists(),'Already complete'
        assert (out/'last.pt').exists(),'No checkpoint; inspect failure before a fresh submission'
        archive=out/f'resume_archive_{time.time_ns()}';archive.mkdir()
        for name in ('launch_receipt.json','FAILED.json','PROCESS_EXIT.json','GPU_STOP.json'):
            path=out/name
            if path.exists():path.rename(archive/name)
        save(dict(group=a.group,time=time.time(),archive=str(archive)),out/'resume_requested.json')
    start()
if __name__=='__main__':main()
