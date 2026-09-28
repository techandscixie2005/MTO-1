#!/usr/bin/env python3
"""Additive GPU2 A/B/C queue after reviewed administrative G2 retirement."""
import fcntl, importlib.util, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path("/home/inspur/MTO-1/experiments/qm9s_chan64_20260928")
PY=Path("/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python")
sys.path.insert(0,str(ROOT))
from train import atomic_json,sha,source_hashes
from g2_retire import process,state_metadata
spec=importlib.util.spec_from_file_location("old_supervisor",OLD/"supervisor.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def status(**kwargs):
    atomic_json({"time":time.time(),"pid":os.getpid(),**kwargs},ROOT/"queue_status.json")
def pid_alive(pid):
    try:
        b=(Path("/proc")/str(pid)/"cmdline").read_bytes()
        return b and b"trainer.py" in b and b"G2" in b
    except (FileNotFoundError,PermissionError):return False
def official_eval_pending():
    all_done=all((OLD/"runs"/n/"FIT_COMPLETE.json").exists() for n in ("G1","G2","G3","G4"))
    return all_done and not ((OLD/"reports/ANALYSIS_COMPLETE.json").exists() or
                              (OLD/"reports/ANALYSIS_FAILED.json").exists())
def admin_verified():
    rp=ROOT/"G2_RETIREMENT_AMENDMENT_REVIEW.json"
    mp=ROOT/"G2_ADMIN_STOPPED.json"
    oldmp=OLD/"runs/G2/ADMIN_STOPPED.json"
    if not rp.exists() or not mp.exists() or not oldmp.exists():
        return False,"amendment review or ADMIN_STOPPED marker missing"
    review=json.loads(rp.read_text())
    if not review.get("passed"):return False,"amendment review failed"
    for name in ("g2_retire.py","pilot_queue_admin.py","pilot_queue.py","train.py","pilot_config.json"):
        if review["source_hashes"].get(str(ROOT/name))!=sha(ROOT/name):
            return False,"amendment source changed: "+name
    marker=json.loads(mp.read_text())
    if marker!=json.loads(oldmp.read_text()) or marker.get("event")!="ADMIN_STOPPED":
        return False,"ADMIN_STOPPED marker mismatch"
    if marker.get("amendment_review_sha256")!=sha(rp) or marker.get("source_hashes")!=review["source_hashes"]:
        return False,"ADMIN_STOPPED amendment binding mismatch"
    for key in ("g2_process_exited","waiter_exited_before_G2_signal",
                "g2_fit_complete_absent","pilot_workers_absent"):
        if marker.get(key) is not True:return False,"ADMIN_STOPPED verification missing: "+key
    receipt=json.loads((OLD/"runs/G2/launch_receipt.json").read_text())
    if marker["g2_pid"]!=receipt["pid"] or marker["gpu"]!=2:
        return False,"G2 receipt mismatch"
    old=OLD/"runs/G2"
    if (old/"FIT_COMPLETE.json").exists():return False,"Unexpected G2 FIT_COMPLETE"
    info=process(receipt["pid"])
    if info and info["start_ticks"]==marker["g2_start_ticks"]:
        return False,"Original G2 process still present"
    if sha(old/"last.pt")!=marker["after"]["last_pt_sha256"]:
        return False,"Retained G2 last checkpoint changed"
    if sha(old/"best.pt")!=marker["after"]["best_pt_sha256"]:
        return False,"Retained G2 best checkpoint changed"
    if state_metadata(old/"last.pt")!=marker["after"]["last_state"]:
        return False,"Retained G2 checkpoint state changed"
    return True,"passed"
def reviewed():
    rp=ROOT/"CODE_REVIEW.json";pp=ROOT/"preflight_results.json"
    if not rp.exists() or not pp.exists():return False,"review/preflight missing"
    r=json.loads(rp.read_text());p=json.loads(pp.read_text())
    if not r.get("passed") or not p.get("passed"):return False,"review/preflight failed"
    for name in ("train.py","preflight.py","pilot_queue.py","summarize.py","pilot_objective.py","pilot_config.json"):
        if r.get(name+"_sha256")!=sha(ROOT/name):return False,"review hash mismatch "+name
    if p["source_hashes"]!=source_hashes():return False,"stale preflight source hash"
    return admin_verified()
def free_clean_gpu2():
    h,busy,xml=old.health()
    g=h[2]
    return bool(g["clean"] and g["uuid"] not in busy and g["memory_MiB"]<=200),h,busy,xml
def foreign_gpu2_process(child_pid):
    lines=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader"],text=True).splitlines()
    h,_,_=old.health();uuid=h[2]["uuid"]
    return [int(bits[1]) for line in lines if len(bits:=[b.strip() for b in line.split(",")])==2
            and bits[0]==uuid and bits[1].isdigit() and int(bits[1])!=child_pid]
def main():
    lock=open(ROOT/"pilot_queue.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/"pilot_config.json").read_text())
    assert cfg["arms"]==["control","weighted","direct_f_matched"]
    receipt=json.loads((OLD/"runs/G2/launch_receipt.json").read_text())
    assert receipt["gpu"]==2
    while not STOP:
        okay,why=reviewed()
        if not okay:
            status(state="WAIT_ADMIN_STOP_AND_REVIEW",reason=why,g2_pid=receipt["pid"])
            time.sleep(30);continue
        break
    if STOP:
        status(state="STOPPED_BEFORE_LAUNCH");return
    for arm in cfg["arms"]:
        out=ROOT/"runs"/arm;out.mkdir(parents=True,exist_ok=True)
        if (out/"FIT_COMPLETE.json").exists():
            fit=json.loads((out/"FIT_COMPLETE.json").read_text())
            assert fit["source_hashes"]==source_hashes()
            continue
        if (out/"FAILED.json").exists():
            raise RuntimeError("Prior arm failed and needs review: "+arm)
        while not STOP:
            okay,why=reviewed()
            if not okay:
                status(state="WAIT_REVIEW",arm=arm,reason=why);time.sleep(30);continue
            if official_eval_pending():
                status(state="WAIT_OFFICIAL_EVALUATION",arm=arm);time.sleep(30);continue
            free,h,busy,xml=free_clean_gpu2()
            if not free:
                status(state="WAIT_HEALTHY_IDLE_GPU2",arm=arm,
                       gpu2=h[2],busy=h[2]["uuid"] in busy);time.sleep(30);continue
            glock=open("/tmp/mto_pouter_gpu_2.lock","w")
            try:fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                glock.close();status(state="WAIT_GPU2_LOCK",arm=arm);time.sleep(30);continue
            free,h,busy,xml=free_clean_gpu2()
            if not free:
                glock.close();time.sleep(30);continue
            break
        if STOP:
            status(state="STOPPED_BEFORE_ARM",arm=arm);return
        logs=ROOT/"logs";logs.mkdir(exist_ok=True)
        (logs/(arm+"_gpu_before.xml")).write_text(xml)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="2",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
                 PYTHONUNBUFFERED="1",PYTHONWARNINGS="ignore",PYTHONDONTWRITEBYTECODE="1")
        with open(logs/(arm+".log"),"a") as logfile:
            child=subprocess.Popen([str(PY),"-u",str(ROOT/"train.py"),arm],cwd=ROOT,
                env=env,stdout=logfile,stderr=subprocess.STDOUT,pass_fds=(glock.fileno(),),
                start_new_session=True)
            atomic_json({"arm":arm,"pid":child.pid,"gpu":2,"gpu_uuid":h[2]["uuid"],
                         "started":time.time(),"health_clean":h[2]["clean"],
                         "memory_before_MiB":h[2]["memory_MiB"],
                         "review_sha256":sha(ROOT/"CODE_REVIEW.json"),
                         "amendment_review_sha256":sha(ROOT/"G2_RETIREMENT_AMENDMENT_REVIEW.json"),
                         "admin_stopped_sha256":sha(ROOT/"G2_ADMIN_STOPPED.json"),
                         "admin_queue_sha256":sha(ROOT/"pilot_queue_admin.py"),
                         "preflight_sha256":sha(ROOT/"preflight_results.json"),
                         "source_hashes":source_hashes()},out/"queue_launch_receipt.json")
            status(state="RUNNING",arm=arm,child_pid=child.pid,gpu=2)
            while True:
                try:code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:pass
                hnow,_,_=old.health()
                if STOP or not hnow[2]["clean"] or foreign_gpu2_process(child.pid):
                    child.send_signal(signal.SIGTERM)
                    code=child.wait()
                    status(state="STOPPED_FOR_HEALTH_OR_CONFLICT",arm=arm,
                           child_pid=child.pid,exit_code=code)
                    glock.close();return
                if official_eval_pending():
                    h1=hnow[1]
                    if not h1["clean"] or h1["memory_MiB"]>200:
                        child.send_signal(signal.SIGTERM)
                        code=child.wait()
                        status(state="STOPPED_FOR_OFFICIAL_EVALUATION",arm=arm,
                               child_pid=child.pid,exit_code=code)
                        glock.close();return
                status(state="RUNNING",arm=arm,child_pid=child.pid,gpu=2,
                       progress=json.loads((out/"status.json").read_text()) if (out/"status.json").exists() else None)
        (logs/(arm+"_gpu_after.xml")).write_text(old.health()[2])
        glock.close()
        if code!=0 or not (out/"FIT_COMPLETE.json").exists():
            status(state="ARM_NEEDS_REVIEW",arm=arm,child_pid=child.pid,exit_code=code)
            return
        status(state="ARM_COMPLETE",arm=arm,child_pid=child.pid)
    status(state="SUMMARIZING")
    with open(ROOT/"logs/summarize.log","a") as f:
        code=subprocess.call([str(PY),str(ROOT/"summarize.py")],cwd=ROOT,
                             env=dict(os.environ,PYTHONWARNINGS="ignore"),stdout=f,stderr=subprocess.STDOUT)
    if code:status(state="ANALYSIS_FAILED",exit_code=code)
    else:status(state="ALL_COMPLETE",summary=str(ROOT/"pilot_summary.json"))
if __name__=="__main__":
    try:main()
    except BlockingIOError:print("Queue already active",flush=True)
    except BaseException as exc:
        status(state="QUEUE_FAILED",error=str(exc),traceback=traceback.format_exc())
        raise

