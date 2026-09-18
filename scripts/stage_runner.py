"""One stage = preflight if needed, all paired fits, evaluation and reports.

Two allocated GPUs run disjoint seed lists; the next Slurm job depends on the
whole stage. Nonzero exit blocks downstream jobs. USR1 checkpoints and requeues
the SAME Slurm job, preserving the original dependency chain.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from protocol import verify
ROOT=Path(__file__).resolve().parents[1]
STOP=False
CHILDREN=[]

def request_stop(signum,frame):
    global STOP
    STOP=True
    for child in CHILDREN:
        if child.poll() is None:child.send_signal(signal.SIGUSR1)

def run_checked(command,env=None):
    child=subprocess.Popen(command,env=env)
    CHILDREN.append(child)
    code=child.wait();CHILDREN.remove(child)
    if STOP:
        # Parent stage may be interrupted during preflight or evaluation too.
        if '--worker' not in sys.argv:
            subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
        raise SystemExit(75)
    if code:raise SystemExit(code)

def worker(args,seeds):
    for seed in seeds[args.worker::2]:
        if STOP:raise SystemExit(75)
        run_checked([sys.executable,str(ROOT/'scripts/train_scale.py'),'--scale',args.scale,'--seed',str(seed)])
    if STOP:raise SystemExit(75)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale',choices=['1k','10k','full'],required=True)
    parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    signal.signal(signal.SIGUSR1,request_stop)
    frozen=verify()
    execution=json.loads((ROOT/'configs/execution.json').read_text());seeds=execution['seeds_by_scale'][args.scale]
    previous={'1k':None,'10k':'1k','full':'10k'}[args.scale]
    if previous:
        marker=ROOT/'results'/previous/'STAGE_COMPLETE.json'
        payload=json.loads(marker.read_text())
        assert payload['protocol_id']==frozen['protocol_id'] and payload['passed']
        assert (ROOT/'results'/previous/'COMPLETE').exists()
    if args.worker is not None:
        assert json.loads((ROOT/'results/preflight.json').read_text())['passed']
        return worker(args,seeds)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    assert len(visible)==2 and all(visible),f'Expected two allocated GPUs, got {visible}'
    first_env={**os.environ,'CUDA_VISIBLE_DEVICES':visible[0]}
    stage_out=ROOT/'results'/args.scale;stage_out.mkdir(exist_ok=True)
    if (stage_out/'STAGE_COMPLETE.json').exists():
        assert json.loads((stage_out/'STAGE_COMPLETE.json').read_text())['protocol_id']==frozen['protocol_id']
        return
    if args.scale=='1k':run_checked([sys.executable,str(ROOT/'scripts/preflight.py')],first_env)
    started=time.time();logs=[]
    for index,device in enumerate(visible):
        log=(ROOT/'logs'/f'{args.scale}_worker{index}.log').open('a',buffering=1);logs.append(log)
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--scale',args.scale,'--worker',str(index)],
                               env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=log,stderr=subprocess.STDOUT)
        CHILDREN.append(child)
    failures=[]
    while CHILDREN:
        for child in CHILDREN[:]:
            code=child.poll()
            if code is not None:
                CHILDREN.remove(child)
                if code:failures.append(code);request_stop(None,None)
        if CHILDREN:time.sleep(2)
    for log in logs:log.close()
    if STOP or failures:
        if failures and any(code!=75 for code in failures):raise SystemExit(1)
        subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
        raise SystemExit(75)
    for seed in seeds:assert (ROOT/'runs'/f'{args.scale}_seed{seed}'/'DONE').exists()
    run_checked([sys.executable,'-u',str(ROOT/'scripts/evaluate_scale.py'),'--scale',args.scale],first_env)
    run_checked([sys.executable,'-u',str(ROOT/'scripts/export_mto_examples.py'),'--scale',args.scale],first_env)
    assert (stage_out/'COMPLETE').exists()
    marker={'passed':True,'scale':args.scale,'seeds':seeds,'protocol_id':frozen['protocol_id'],
            'job_id':os.environ['SLURM_JOB_ID'],'elapsed_seconds':time.time()-started,
            'completed_unix':time.time(),'previous':previous}
    temporary=stage_out/'STAGE_COMPLETE.json.tmp';temporary.write_text(json.dumps(marker,indent=2))
    temporary.replace(stage_out/'STAGE_COMPLETE.json')
    if args.scale=='full':
        (ROOT/'results/COMPLETE').write_text('All 26 full-architecture fits, evaluations, reports and atomic/tensor exports completed in strict 1k -> 10k -> full order.\n')
    print('STAGE COMPLETE',json.dumps(marker),flush=True)

if __name__=='__main__':main()
