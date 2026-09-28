#!/usr/bin/env python3
"""Launch one reviewed eta0 seed worker after the Gram GPU lock is released."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,ARCH,source_seal,code_hashes,sha,save_json
GRAM=ROOT.parent/"frozen_gram_probe"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("seed",type=int,choices=[23,37])
    ap.add_argument("--gpu",type=int,choices=[1,6],required=True);a=ap.parse_args()
    out=ROOT/f"runs/seed{a.seed}";out.mkdir(parents=True,exist_ok=True)
    assert not (out/"LAUNCH_RECEIPT.json").exists()
    assert not (out/"FIT_COMPLETE.json").exists()
    assert not (out/"FAILED.json").exists() and not (out/"INVALID.json").exists()
    # Gram scientific success is not a prerequisite for this independent seed study.
    # Its current or preserved prior worker must have terminated and released its GPU lock.
    gram_receipt_path=(GRAM/"LAUNCH_RECEIPT.json") if (GRAM/"LAUNCH_RECEIPT.json").exists() else (GRAM/"LAUNCH_RECEIPT_ATTEMPT1_REF.json")
    assert gram_receipt_path.exists()
    gram_receipt=json.loads(gram_receipt_path.read_text())
    gram_gpu=int(gram_receipt["gpu"])
    terminal=any((GRAM/name).exists() for name in
        ("PROBE_COMPLETE.json","FAILED.json","INVALID.json"))
    if gram_receipt_path.name=="LAUNCH_RECEIPT_ATTEMPT1_REF.json":
        terminal=terminal or (GRAM/"FAILED_ATTEMPT1_REF.json").exists()
    assert terminal,"Gram worker has no terminal record"
    proc=Path(f"/proc/{gram_receipt['pid']}/stat")
    if proc.exists():
        fields=proc.read_text().split()
        assert int(fields[21])!=gram_receipt["start_ticks"] or fields[2]=="Z","Gram worker still active"
    gram_lock=open(f"/tmp/mto_pouter_gpu_{gram_gpu}.lock","a+")
    fcntl.flock(gram_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(gram_lock,fcntl.LOCK_UN);gram_lock.close()
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"]
    assert pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    assert pre["initialization"][str(a.seed)]["initial_pt_sha256"]==sha(out/"initial.pt")
    lock=open(f"/tmp/mto_pouter_gpu_{a.gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH));import gpu_health
    before=gpu_health.snapshot(a.gpu)
    assert gpu_health.eligible(before)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),MTO_PHYSICAL_GPU=str(a.gpu),
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",PYTHONWARNINGS="ignore")
    log=open(out/"train.log","a",buffering=1)
    cmd=["/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python",
        str(ROOT/"train.py"),str(a.seed)]
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
    ticks=int(Path(f"/proc/{child.pid}/stat").read_text().split()[21])
    receipt={"classification":"One fresh legacy eta0 seed100 run",
        "seed":a.seed,"pid":child.pid,"start_ticks":ticks,"cwd":str(ROOT),
        "cmd":cmd,"gpu":a.gpu,"gpu_uuid":before["uuid"],"health_before":before,
        "source_hashes":source,"code_hashes":code,
        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),
        "review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),
        "gram_terminal_record":str(GRAM/"PROBE_COMPLETE.json" if (GRAM/"PROBE_COMPLETE.json").exists() else
             (GRAM/"FAILED.json" if (GRAM/"FAILED.json").exists() else GRAM/"FAILED_ATTEMPT1_REF.json")),
        "gram_receipt_sha256":sha(gram_receipt_path),
        "started":time.time(),"log":str(out/"train.log")}
    save_json(receipt,out/"LAUNCH_RECEIPT.json")
    time.sleep(3)
    assert child.poll() is None,f"Worker exited immediately with {child.returncode}"
    print(json.dumps({"state":"RUNNING","seed":a.seed,"pid":child.pid,
        "start_ticks":ticks,"gpu":a.gpu,"receipt":str(out/"LAUNCH_RECEIPT.json")}),flush=True)
if __name__=="__main__":main()

