#!/usr/bin/env python3
"""Concurrent guarded architecture queue; leaves existing campaigns untouched."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path("/home/inspur/MTO-1/experiments/qm9s_chan64_20260928")
PY=Path("/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python")
sys.path.insert(0,str(ROOT))
from gpu_health import snapshot,eligible,unchanged
from train import atomic_json,sha,source_hashes
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def status(**kwargs):
    atomic_json({"time":time.time(),"pid":os.getpid(),**kwargs},ROOT/"ARCH_QUEUE_STATUS.json")
def pid_running(pid,arm,script="train.py"):
    try:
        c=(Path("/proc")/str(pid)/"cmdline").read_bytes()
        return script.encode() in c and arm.encode() in c
    except (FileNotFoundError,PermissionError):return False
def official_done_or_not_pending():
    allfit=all((OLD/"runs"/n/"FIT_COMPLETE.json").exists() for n in ("G1","G2","G3","G4"))
    return not allfit or (OLD/"reports/ANALYSIS_COMPLETE.json").exists() or (OLD/"reports/ANALYSIS_FAILED.json").exists()
def approved():
    pp=ROOT/"PREFLIGHT.json";rp=ROOT/"EXECUTION_REVIEW.json"
    if not pp.exists() or not rp.exists():return False,"preflight/review pending"
    pre=json.loads(pp.read_text());rev=json.loads(rp.read_text())
    if not pre.get("passed") or not rev.get("passed"):return False,"preflight/review failed"
    hashes=source_hashes()
    if pre.get("source_hashes")!=hashes or rev.get("source_hashes")!=hashes:return False,"stale source hashes"
    return True,"passed"
def old_gpu_released(gpu):
    owner={1:"G1",4:"G3",6:"G4"}[gpu]
    out=OLD/"runs"/owner
    if not (out/"FIT_COMPLETE.json").exists():return False
    receipt=json.loads((out/"launch_receipt.json").read_text())
    return not pid_running(receipt["pid"],owner,"trainer.py")
def foreign_apps(row,pid):
    return [p for p in row["apps"] if p!=pid]
def main():
    lock=open(ROOT/"architecture_queue.lock","w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/"config.json").read_text())
    assert cfg["arms"]==["original","retained_residual","direct_f","independent_trace"]
    active={};finished=set();failed=set();excluded_gpu=set();stop_sent=set()
    schedule={"original":[1,4,6,5],"retained_residual":[5,1,4,6],
              "direct_f":[1,4,6,5],"independent_trace":[1,4,6,5]}
    while True:
        for arm in cfg["arms"]:
            out=ROOT/"runs"/arm
            if (out/"FIT_COMPLETE.json").exists():
                fit=json.loads((out/"FIT_COMPLETE.json").read_text())
                if fit.get("arm")==arm and fit.get("source_hashes")==source_hashes():finished.add(arm)
            if (out/"FAILED.json").exists() or (out/"INVALID.json").exists():failed.add(arm)
        if STOP:
            for arm,item in active.items():
                if arm not in stop_sent:
                    item["proc"].send_signal(signal.SIGTERM)
                    stop_sent.add(arm)
                    atomic_json({"event":"QUEUE_STOP_REQUESTED","arm":arm,
                        "pid":item["proc"].pid,"time":time.time()},
                        ROOT/"runs"/arm/"QUEUE_STOP_REQUESTED.json")
        if STOP and not active:
            status(state="STOPPED",finished=sorted(finished),failed=sorted(failed));return
        okay,why=approved()
        if not okay:
            status(state="WAIT_REVIEW",reason=why,finished=sorted(finished))
            time.sleep(30);continue
        for arm in cfg["arms"]:
            if arm in finished or arm in active or arm in failed or STOP or failed:continue
            out=ROOT/"runs"/arm;out.mkdir(parents=True,exist_ok=True)
            if (out/"queue_launch_receipt.json").exists():
                receipt=json.loads((out/"queue_launch_receipt.json").read_text())
                pid=receipt.get("pid")
                if pid and pid_running(pid,arm):
                    status(state="EXISTING_WORKER_REQUIRES_ORIGINAL_QUEUE",arm=arm,pid=pid)
                    return
                if not (out/"last.pt").exists():
                    failed.add(arm);continue
                # Preserve stopped checkpoint for an explicit resume decision.
                status(state="STOPPED_ARM_REQUIRES_REVIEW",arm=arm)
                return
            if not official_done_or_not_pending():continue
            for gpu in schedule[arm]:
                if gpu in excluded_gpu or any(a["gpu"]==gpu for a in active.values()):continue
                if arm=="original" and gpu==5 and "retained_residual" not in active and "retained_residual" not in finished:continue
                if gpu!=5 and not old_gpu_released(gpu):continue
                row=snapshot(gpu)
                if not eligible(row,guarded=(gpu==5)):continue
                glock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","w")
                try:fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:glock.close();continue
                row2=snapshot(gpu)
                if not eligible(row2,guarded=(gpu==5)) or row2["uuid"]!=row["uuid"]:
                    glock.close();continue
                micro=None
                if gpu==5:
                    micro_env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONWARNINGS="ignore",
                        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        PYTHONDONTWRITEBYTECODE="1")
                    try:
                        check=subprocess.run([str(PY),
                            "/home/inspur/MTO-1/research/oscillator_r2_20260928/resource_admission.py",
                            "--microcheck"],env=micro_env,text=True,capture_output=True,timeout=120)
                        lines=[v for v in check.stdout.splitlines() if v.startswith("{")]
                        micro=json.loads(lines[-1]) if check.returncode==0 and lines else {"passed":False,
                            "exit_code":check.returncode,"stderr_tail":check.stderr[-1500:]}
                    except BaseException as exc:micro={"passed":False,"error":str(exc)}
                    checked=snapshot(gpu)
                    if not micro.get("passed") or not unchanged(row2,checked) or not eligible(checked,guarded=True):
                        excluded_gpu.add(gpu)
                        atomic_json({"gpu":gpu,"numerical_microcheck":micro,"health_before":row2,
                            "health_after":checked,"time":time.time()},ROOT/"GPU5_EXCLUDED.json")
                        glock.close()
                        continue
                    row2=checked
                logs=ROOT/"logs";logs.mkdir(exist_ok=True)
                before={"arm":arm,"gpu":gpu,"gpu_uuid":row2["uuid"],"health_before":row2,
                        "numerical_microcheck":micro,
                        "started":time.time(),"pid":None,
                        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),
                        "review_sha256":sha(ROOT/"EXECUTION_REVIEW.json")}
                atomic_json(before,out/"queue_launch_receipt.json")
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MTO_PHYSICAL_GPU=str(gpu),
                    OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1",
                    PYTHONWARNINGS="ignore",PYTHONDONTWRITEBYTECODE="1")
                logfile=open(logs/(arm+".log"),"a")
                child=subprocess.Popen([str(PY),"-u",str(ROOT/"train.py"),arm],cwd=ROOT,
                    env=env,stdout=logfile,stderr=subprocess.STDOUT,pass_fds=(glock.fileno(),),
                    start_new_session=True)
                before["pid"]=child.pid
                atomic_json(before,out/"queue_launch_receipt.json")
                active[arm]={"proc":child,"gpu":gpu,"baseline":row2,"lock":glock,"log":logfile}
                status(state="RUNNING",active={k:{"pid":v["proc"].pid,"gpu":v["gpu"]} for k,v in active.items()},
                       finished=sorted(finished))
                break
        for arm,item in list(active.items()):
            child=item["proc"];gpu=item["gpu"]
            try:
                now=snapshot(gpu)
                if not unchanged(item["baseline"],now) or foreign_apps(now,child.pid):
                    atomic_json({"event":"INVALID_GPU_HEALTH_OR_FOREIGN_PROCESS",
                        "gpu":gpu,"before":item["baseline"],"observed":now,"time":time.time()},
                        ROOT/"runs"/arm/"INVALID.json")
                    child.send_signal(signal.SIGTERM)
            except BaseException as exc:
                atomic_json({"event":"GPU_HEALTH_CHECK_FAILED","gpu":gpu,"error":str(exc),
                             "time":time.time()},ROOT/"runs"/arm/"INVALID.json")
                child.send_signal(signal.SIGTERM)
            code=child.poll()
            if code is None:continue
            item["log"].close()
            after=snapshot(gpu)
            atomic_json(after,ROOT/"runs"/arm/"gpu_health_after.json")
            item["lock"].close()
            del active[arm]
            if code==0 and (ROOT/"runs"/arm/"FIT_COMPLETE.json").exists() and not (ROOT/"runs"/arm/"INVALID.json").exists():
                finished.add(arm)
            else:failed.add(arm)
        if len(finished)==len(cfg["arms"]):
            status(state="SUMMARIZING",finished=sorted(finished))
            with open(ROOT/"logs/summarize.log","a") as f:
                code=subprocess.call([str(PY),str(ROOT/"summarize.py")],cwd=ROOT,
                    env=dict(os.environ,PYTHONWARNINGS="ignore"),stdout=f,stderr=subprocess.STDOUT)
            status(state="ALL_COMPLETE" if code==0 else "ANALYSIS_FAILED",exit_code=code,
                   finished=sorted(finished))
            return
        if failed and not active:
            status(state="NEEDS_REVIEW",failed=sorted(failed),finished=sorted(finished));return
        status(state="RUNNING" if active else "WAIT_RESOURCES",
               active={k:{"pid":v["proc"].pid,"gpu":v["gpu"]} for k,v in active.items()},
               finished=sorted(finished),failed=sorted(failed))
        time.sleep(20)
if __name__=="__main__":
    try:main()
    except BlockingIOError:print("Architecture queue already active",flush=True)
    except BaseException as exc:
        status(state="QUEUE_FAILED",error=str(exc),traceback=traceback.format_exc())
        raise

