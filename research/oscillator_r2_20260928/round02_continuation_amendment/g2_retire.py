#!/usr/bin/env python3
"""Narrow, identity-checked administrative G2 retirement. Never creates FIT_COMPLETE."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
OLD=Path("/home/inspur/MTO-1/experiments/qm9s_chan64_20260928")
RUN=OLD/"runs/G2"
ARMS=("control","weighted","direct_f_matched")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()
def atomic_json(obj,p):
    p=Path(p);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    os.replace(tmp,p)
def process(pid):
    proc=Path("/proc")/str(pid)
    try:
        raw=(proc/"cmdline").read_bytes()
        cmd=[x.decode(errors="replace") for x in raw.split(b"\0") if x]
        stat=(proc/"stat").read_text()
        start_ticks=int(stat.rsplit(")",1)[1].split()[19])
        cwd=str((proc/"cwd").resolve())
        status=(proc/"status").read_text().splitlines()[2]
        return {"pid":pid,"cmdline":cmd,"cwd":cwd,"start_ticks":start_ticks,"status":status}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def exact_worker(info,receipt):
    return info is not None and info["pid"]==receipt["pid"] and info["cwd"]==str(OLD) and info["cmdline"][-2:]==["trainer.py","G2"]
def exact_waiter(info,status):
    return info is not None and info["pid"]==status["pid"] and any(
        x==str(ROOT/"pilot_queue.py") for x in info["cmdline"])
def pilot_workers():
    found=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():continue
        p=process(int(entry.name))
        if p is None:continue
        if str(ROOT/"train.py") in p["cmdline"] and any(a in p["cmdline"] for a in ARMS):
            found.append(p)
    return found
def state_metadata(path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    s=ck["state"]
    for k in ("rng_numpy","rng_torch","rng_cuda","rng_python","rng_numpy_global"):
        assert k in ck,"Missing resumable RNG: "+k
    assert "optimizer" in ck and "scheduler" in ck and "model" in ck
    cfg=json.loads((OLD/"configs/G2.json").read_text())
    assert ck["config"]==cfg
    manifest=json.loads((RUN/"run_manifest.json").read_text())
    assert ck["fingerprint"]==manifest["fingerprint"]
    return {k:s[k] for k in ("epoch","cursor","steps","best","best_epoch","bad_epochs",
                             "lr_reductions","last_reduction","total_seconds")} | {
        "history_count":len(s["history"]),
        "history_last_epoch":s["history"][-1]["epoch"] if s["history"] else None,
        "has_order":s["order"] is not None,
        "resumable_rng_optimizer_scheduler":True}
def best_metadata(path):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    cfg=json.loads((OLD/"configs/G2.json").read_text())
    manifest=json.loads((RUN/"run_manifest.json").read_text())
    assert ck["config"]==cfg and ck["fingerprint"]==manifest["fingerprint"]
    assert ck["epoch"]==41 and len(ck["val"])>=1 and "model" in ck
    return {"epoch":ck["epoch"],"validation_objective":float(ck["val"][0]),
            "config_fingerprint_match":True}
def architecture_seal():
    ar=ROOT/"architecture"
    pre=json.loads((ar/"PREFLIGHT.json").read_text())
    review=json.loads((ar/"EXECUTION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"] and pre["source_hashes"]==review["source_hashes"]
    for name,digest in pre["source_hashes"].items():
        assert sha(name)==digest,"Running architecture source changed: "+name
    return {"preflight_sha256":sha(ar/"PREFLIGHT.json"),
            "execution_review_sha256":sha(ar/"EXECUTION_REVIEW.json"),
            "source_file_count":len(pre["source_hashes"])}
def current():
    arch=architecture_seal()
    receipt=json.loads((RUN/"launch_receipt.json").read_text())
    status=json.loads((ROOT/"queue_status.json").read_text())
    assert receipt["name"]=="G2" and receipt["gpu"]==2
    assert status["state"]=="WAIT_G2_FINISH" and status["g2_pid"]==receipt["pid"]
    assert time.time()-status["time"]<120,"Old queue status stale"
    assert not (RUN/"FIT_COMPLETE.json").exists()
    assert not (ROOT/"G2_ADMIN_STOPPED.json").exists()
    assert not (RUN/"ADMIN_STOPPED.json").exists()
    assert not pilot_workers(),"Pilot worker already active"
    waiter=process(status["pid"]);worker=process(receipt["pid"])
    assert exact_waiter(waiter,status),"Old waiting queue identity mismatch"
    assert exact_worker(worker,receipt),"G2 worker identity mismatch"
    boot=int(next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines()
                  if line.startswith("btime ")))
    worker_start_wall=boot+worker["start_ticks"]/os.sysconf("SC_CLK_TCK")
    assert abs(worker_start_wall-receipt["time"])<30,"G2 receipt/start identity mismatch"
    worker["start_wall_time"]=worker_start_wall
    assert worker["start_ticks"]<waiter["start_ticks"]
    assert sha(ROOT/"pilot_queue.py")==json.loads((ROOT/"CODE_REVIEW.json").read_text())["pilot_queue.py_sha256"]
    before=state_metadata(RUN/"last.pt")
    best=best_metadata(RUN/"best.pt")
    assert before["best"]==best["validation_objective"]
    assert before["best_epoch"]==best["epoch"]==41 and before["epoch"]>=210 and before["bad_epochs"]>=160
    assert before["lr_reductions"]>=3
    return {"time":time.time(),"reason":"G2 validation plateau after epoch41; third LR reduction at194 delays natural stop to at least244; native-f deployable score unavailable for unsupervised E",
            "receipt":receipt,"receipt_sha256":sha(RUN/"launch_receipt.json"),
            "old_waiter":waiter,"g2_worker":worker,"old_queue_status":status,
            "before":{"last_pt_sha256":sha(RUN/"last.pt"),"best_pt_sha256":sha(RUN/"best.pt"),
                       "last_state":before,"best_checkpoint":best,
                      "status":json.loads((RUN/"status.json").read_text())},
            "pilot_workers":[],"architecture_seal":arch}
def wait_exit(pid,start_ticks,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        p=process(pid)
        if p is None or p["start_ticks"]!=start_ticks:return True
        if "Z (zombie)" in p["status"]:return True
        time.sleep(1)
    return False
def execute():
    review_path=ROOT/"G2_RETIREMENT_AMENDMENT_REVIEW.json"
    review=json.loads(review_path.read_text())
    assert review["passed"]
    for name in ("g2_retire.py","pilot_queue_admin.py"):
        assert review["source_hashes"][str(ROOT/name)]==sha(ROOT/name),"Stale amendment review"
    for name in ("pilot_queue.py","train.py","pilot_config.json"):
        assert review["source_hashes"][str(ROOT/name)]==sha(ROOT/name),"Frozen source changed"
    pre=current()
    atomic_json(pre,ROOT/"G2_RETIREMENT_PREFLIGHT.json")
    waiter=pre["old_waiter"];worker=pre["g2_worker"]
    assert exact_waiter(process(waiter["pid"]),pre["old_queue_status"])
    assert process(waiter["pid"])["start_ticks"]==waiter["start_ticks"]
    os.kill(waiter["pid"],signal.SIGTERM)
    assert wait_exit(waiter["pid"],waiter["start_ticks"],60),"Old waiting queue did not exit"
    assert not pilot_workers(),"Pilot worker appeared after old queue stop"
    fresh=process(worker["pid"])
    assert exact_worker(fresh,pre["receipt"]) and fresh["start_ticks"]==worker["start_ticks"]
    os.kill(worker["pid"],signal.SIGTERM)
    assert wait_exit(worker["pid"],worker["start_ticks"],180),"G2 did not stop gracefully"
    status=json.loads((RUN/"status.json").read_text())
    after=state_metadata(RUN/"last.pt")
    if status["event"]=="STOPPED_CHECKPOINTED":
        assert after["epoch"]==status["epoch"] and after["cursor"]==status["cursor"]
        stop_boundary="within_epoch"
    elif status["event"]=="EPOCH_COMPLETE":
        assert after["epoch"]==status["epoch"]+1 and after["cursor"]==0
        assert not after["has_order"] and after["history_last_epoch"]==status["epoch"]
        assert after["steps"]==status["steps"]
        stop_boundary="after_epoch_checkpoint"
    else:
        raise AssertionError("G2 lacks a verified graceful checkpoint")
    assert after["best_epoch"]==41 and after["epoch"]>=pre["before"]["last_state"]["epoch"]
    assert after["steps"]>=pre["before"]["last_state"]["steps"]
    assert sha(RUN/"best.pt")==pre["before"]["best_pt_sha256"],"Best checkpoint changed"
    assert best_metadata(RUN/"best.pt")==pre["before"]["best_checkpoint"]
    assert after["best"]==pre["before"]["best_checkpoint"]["validation_objective"]
    assert not (RUN/"FIT_COMPLETE.json").exists()
    marker={"event":"ADMIN_STOPPED","classification":"intentional incomplete G2 training",
        "time":time.time(),"reason":pre["reason"],"g2_pid":worker["pid"],
        "g2_start_ticks":worker["start_ticks"],"g2_process_exited":True,
        "waiter_pid":waiter["pid"],"waiter_start_ticks":waiter["start_ticks"],
        "waiter_exited_before_G2_signal":True,"gpu":2,"gpu_uuid":pre["receipt"]["gpu_uuid"],
        "before":pre["before"],"after":{"last_pt_sha256":sha(RUN/"last.pt"),
            "best_pt_sha256":sha(RUN/"best.pt"),"last_state":after,"status":status},
        "g2_fit_complete_absent":True,"pilot_workers_absent":not bool(pilot_workers()),
        "stop_boundary":stop_boundary,"architecture_seal_before":pre["architecture_seal"],
        "architecture_seal_after":architecture_seal(),
        "source_hashes":review["source_hashes"],"amendment_review_sha256":sha(review_path)}
    assert marker["pilot_workers_absent"]
    atomic_json(marker,RUN/"ADMIN_STOPPED.json")
    atomic_json(marker,ROOT/"G2_ADMIN_STOPPED.json")
    md=f"""# Administrative retirement of G2

G2 was intentionally stopped on {time.strftime('%Y-%m-%d %H:%M:%S %Z',time.localtime(marker['time']))}. This is an incomplete run, not a natural FIT_COMPLETE result.

**Decision.** The best validation objective remained at epoch 41 through the active epoch {after['epoch']}. The third learning-rate reduction at epoch 194 moves the earliest natural stopping point to epoch 244, so continuing would occupy GPU2 for a low-information interval. G2 cannot yield deployable native-f predictions because E was unsupervised.

The original waiting loss-pilot queue PID {waiter['pid']} exited before G2 received SIGTERM. No pilot worker was active. The exact G2 process PID {worker['pid']} with start tick {worker['start_ticks']}, working directory {worker['cwd']}, and GPU2 receipt was rechecked immediately before signaling. The existing trainer handled SIGTERM and exited at the verified `{stop_boundary}` checkpoint boundary (actual status: `{status['event']}`). The original supervisor may later classify the four-group campaign as blocked; that status is expected and must not be overwritten.

| State | Epoch | Cursor | Steps | Best epoch | LR reductions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | {pre['before']['last_state']['epoch']} | {pre['before']['last_state']['cursor']} | {pre['before']['last_state']['steps']} | {pre['before']['last_state']['best_epoch']} | {pre['before']['last_state']['lr_reductions']} |
| After | {after['epoch']} | {after['cursor']} | {after['steps']} | {after['best_epoch']} | {after['lr_reductions']} |

The resumable last checkpoint, optimizer, scheduler, all RNG states, and unchanged best checkpoint remain on the server. Their SHA-256 digests and process identity are in G2_ADMIN_STOPPED.json. No FIT_COMPLETE marker was created, and no GPU reset occurred. The replacement loss-pilot queue may use GPU2 only after independent idle/health/lock checks.
"""
    (ROOT/"G2_RETIREMENT.md").write_text(md)
    print(json.dumps({"admin_stopped":True,"g2_pid":worker["pid"],"after":after,
        "last_sha256":marker["after"]["last_pt_sha256"],"best_sha256":marker["after"]["best_pt_sha256"]}))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("preflight","execute"));args=ap.parse_args()
    if args.mode=="preflight":
        p=current();atomic_json(p,ROOT/"G2_RETIREMENT_PREFLIGHT.json")
        print(json.dumps({"passed":True,"g2_pid":p["g2_worker"]["pid"],
            "epoch":p["before"]["last_state"]["epoch"],"best_epoch":p["before"]["last_state"]["best_epoch"],
            "waiter_pid":p["old_waiter"]["pid"]}))
    else:execute()
if __name__=="__main__":main()

