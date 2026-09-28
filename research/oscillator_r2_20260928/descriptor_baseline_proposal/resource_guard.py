"""Admission and hard supervised process envelope for one future CPU descriptor run."""
from __future__ import annotations
import json,os,resource,signal,subprocess,time
from pathlib import Path
import psutil

LIMIT_BYTES=16*1024**3
WALL_SECONDS=4*3600
CPU_LIMIT_SECONDS=32*3600
WORKERS=8

def admit():
    proc=psutil.Process()
    affinity=proc.cpu_affinity()
    per_cpu=psutil.cpu_percent(interval=0.2,percpu=True)
    idle=sorted((i for i in affinity if per_cpu[i]<=50.0),key=lambda i:(per_cpu[i],i))
    available=psutil.virtual_memory().available
    if len(affinity)<WORKERS or len(idle)<WORKERS:
        raise RuntimeError("fewer than eight sufficiently idle admitted CPUs")
    if available<LIMIT_BYTES+4*1024**3:
        raise RuntimeError("less than 16 GiB process budget plus 4 GiB headroom")
    return {"allowed_logical_cpus":len(affinity),"idle_below_50_percent":len(idle),
            "selected_cpus":idle[:WORKERS],"selected_percent_busy":[per_cpu[i] for i in idle[:WORKERS]],
            "available_bytes":available,"process_limit_bytes":LIMIT_BYTES,
            "wall_limit_seconds":WALL_SECONDS,"fit_cpu_limit_seconds":CPU_LIMIT_SECONDS}

def _tree_usage(proc):
    rss=cpu=0.0
    try: family=[proc,*proc.children(recursive=True)]
    except psutil.NoSuchProcess: family=[]
    for p in family:
        try:
            rss+=p.memory_info().rss
            t=p.cpu_times();cpu+=t.user+t.system
        except psutil.NoSuchProcess:pass
    return int(rss),float(cpu)

def supervised_fit(command,cwd,receipt_path,env,log_path,budget_seconds,
                   limit_bytes=LIMIT_BYTES,cpu_limit_seconds=None,admission=None):
    """Run one phase with RLIMIT_AS and RSS/CPU/wall watchdog; record resource exits."""
    if not (0<budget_seconds<=WALL_SECONDS and 0<limit_bytes<=LIMIT_BYTES):
        raise RuntimeError("invalid resource envelope")
    admission=admission if admission is not None else admit()
    selected=admission["selected_cpus"]
    def child_setup():
        os.sched_setaffinity(0,set(selected))
        resource.setrlimit(resource.RLIMIT_AS,(limit_bytes,limit_bytes))
    start=time.monotonic();peak_rss=peak_cpu=0
    with open(log_path,"ab",buffering=0) as log:
        child=subprocess.Popen(command,cwd=cwd,env=env,start_new_session=True,
                               stdout=log,stderr=subprocess.STDOUT,preexec_fn=child_setup)
        proc=psutil.Process(child.pid);reason=None
        while child.poll() is None:
            rss,cpu=_tree_usage(proc)
            peak_rss=max(peak_rss,rss);peak_cpu=max(peak_cpu,cpu)
            elapsed=time.monotonic()-start
            if peak_rss>limit_bytes:reason="process_RSS_over_limit"
            elif elapsed>budget_seconds:reason="wall_clock_over_total_budget"
            elif cpu_limit_seconds is not None and peak_cpu>cpu_limit_seconds:
                reason="fit_CPU_core_seconds_over_32h"
            if reason:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(min(1,max(0.05,budget_seconds/4)))
        child.wait()
    if reason is None and child.returncode!=0:
        tail=Path(log_path).read_bytes()[-65536:].decode("utf-8",errors="replace").lower()
        tokens=("memoryerror","cannot allocate memory","unable to allocate","std::bad_alloc",
                "out of memory","errno 12","failed to map segment")
        if child.returncode==86 or any(x in tail for x in tokens):reason="address_space_or_allocation_failure"
    status={"child_pid":child.pid,"returncode":child.returncode,
            "elapsed_seconds":time.monotonic()-start,"peak_tree_rss_bytes":peak_rss,
            "peak_tree_cpu_seconds":peak_cpu,"resource_limit_bytes":limit_bytes,
            "phase_wall_budget_seconds":budget_seconds,"phase_cpu_limit_seconds":cpu_limit_seconds,
            "admission":admission,"resource_failure":reason}
    if reason:
        path=Path(receipt_path)
        path.write_text(json.dumps({"event":"INCOMPLETE_RESOURCE",**status},indent=2)+"\n",encoding="utf-8")
    return status

