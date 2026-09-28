#!/usr/bin/env python3
"""Detached launch of the one reviewed frozen-head run after preflight."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,ARCH,sha,atomic_json,config
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu",type=int,required=True);args=ap.parse_args()
    cfg=config();assert args.gpu in cfg["allowed_physical_gpus"]
    assert not (ROOT/"run/FIT_COMPLETE.json").exists()
    assert not (ROOT/"run/queue_launch_receipt.json").exists()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    sys.path.insert(0,str(ROOT))
    from train import current_hashes
    assert pre["passed"] and review["passed"] and review["source_hashes"]==current_hashes()
    assert pre["train_script_sha256"]==sha(ROOT/"train.py")
    sys.path.insert(0,str(ARCH))
    from gpu_health import snapshot,eligible
    before=snapshot(args.gpu)
    assert eligible(before,guarded=args.gpu==5)
    if args.gpu==5:raise RuntimeError("GPU5 requires separate numerical microcheck")
    out=ROOT/"run";out.mkdir(exist_ok=True)
    log=open(out/"train.log","a",buffering=1)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),MTO_PHYSICAL_GPU=str(args.gpu),
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    cmd=["/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python",
         str(ROOT/"train.py")]
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
    receipt={"classification":"Frozen residual head only, single authorized20epoch run",
        "pid":child.pid,"gpu":args.gpu,"gpu_uuid":before["uuid"],"health_before":before,
        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),
        "review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),
        "train_script_sha256":sha(ROOT/"train.py"),"started":time.time(),
        "log":str(out/"train.log")}
    atomic_json(receipt,out/"queue_launch_receipt.json")
    time.sleep(3)
    code=child.poll()
    assert code is None,f"Training child exited immediately, code {code}"
    print(json.dumps({"state":"TRAINING","pid":child.pid,"gpu":args.gpu,
        "receipt":str(out/"queue_launch_receipt.json"),"log":str(out/"train.log")}))
if __name__=="__main__":main()

