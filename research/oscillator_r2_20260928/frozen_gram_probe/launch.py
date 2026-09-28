#!/usr/bin/env python3
"""Launch exactly one reviewed frozen Gram probe on an admitted clean GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from features import ROOT,ARCH,sha,save_json,cfg,source_seal,code_hashes
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu",type=int,required=True);a=ap.parse_args()
    c=cfg();assert a.gpu in c["allowed_gpu_clean"]
    assert not (ROOT/"LAUNCH_RECEIPT.json").exists()
    assert not (ROOT/"PROBE_COMPLETE.json").exists()
    assert not (ROOT/"FAILED.json").exists() and not (ROOT/"INVALID.json").exists()
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"]
    assert pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    sys.path.insert(0,str(ARCH))
    import gpu_health
    before=gpu_health.snapshot(a.gpu)
    assert gpu_health.eligible(before)
    log=open(ROOT/"run_attempt2.log","a",buffering=1)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),
        MTO_PHYSICAL_GPU=str(a.gpu),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1",PYTHONWARNINGS="ignore")
    cmd=["/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python",
         str(ROOT/"run.py")]
    worker=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
    start_ticks=int(Path(f"/proc/{worker.pid}/stat").read_text().split()[21])
    receipt={"classification":"One authorized fixed-lambda two-arm frozen Gram probe",
        "started":time.time(),"pid":worker.pid,"start_ticks":start_ticks,"gpu":a.gpu,"gpu_uuid":before["uuid"],
        "health_before":before,"cwd":str(ROOT),"command":cmd,
        "source_hashes":source,"code_hashes":code,
        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),
        "review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),
        "log":str(ROOT/"run_attempt2.log")}
    save_json(receipt,ROOT/"LAUNCH_RECEIPT.json")
    time.sleep(3)
    assert worker.poll() is None,f"Worker exited immediately with {worker.returncode}"
    print(json.dumps({"state":"RUNNING","pid":worker.pid,"gpu":a.gpu,
        "receipt":str(ROOT/"LAUNCH_RECEIPT.json"),"log":str(ROOT/"run_attempt2.log")}))
if __name__=="__main__":main()

