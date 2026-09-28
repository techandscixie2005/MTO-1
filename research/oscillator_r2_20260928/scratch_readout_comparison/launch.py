#!/usr/bin/env python3
"""Launch one reviewed scratch arm on an independently healthy physical GPU."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,source_seal,code_hashes,sha,save_json
def pid_live(receipt):
    proc=Path(f"/proc/{receipt['pid']}/stat")
    if not proc.exists():return False
    parts=proc.read_text().split()
    return parts[2]!="Z" and int(parts[21])==receipt["start_ticks"]
def main():
    a=argparse.ArgumentParser();a.add_argument("arm",choices=["native","mto"])
    a.add_argument("--gpu",required=True,type=int,choices=[2,4,5]);v=a.parse_args()
    out=ROOT/"runs"/v.arm;out.mkdir(parents=True,exist_ok=True)
    assert not any((out/n).exists() for n in ("LAUNCH_RECEIPT.json","FIT_COMPLETE.json","FAILED.json","INVALID.json"))
    assert not (out/"last.pt").exists()
    # The corrected Gram worker used physical GPU2; verify its recorded terminal failure
    # and process exit before reusing that device. Scientific Gram computation was recovered.
    if v.gpu==2:
        gram=ROOT.parent/"frozen_gram_probe"
        receipt=json.loads((gram/"LAUNCH_RECEIPT.json").read_text())
        assert receipt["gpu"]==2 and not pid_live(receipt)
        assert (gram/"FAILED.json").exists() and (gram/"POSTPROCESS_RECOVERY.json").exists()
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"]
    assert pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    lock=open(f"/tmp/mto_pouter_gpu_{v.gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ROOT.parent/"architecture"));import gpu_health
    before=gpu_health.snapshot(v.gpu)
    assert gpu_health.eligible(before,guarded=(v.gpu==5)),before
    if v.gpu==5:
        admit=json.loads((ROOT/"GPU5_ADMISSION.json").read_text())
        review5=json.loads((ROOT/"GPU5_ADMISSION_REVIEW.json").read_text())
        assert admit["passed"] and admit["microcheck_passed"] and admit["gpu_uuid"]==before["uuid"]
        assert admit["snapshot"]["ecc"]==before["ecc"]
        assert review5["passed"] and review5["admission_sha256"]==sha(ROOT/"GPU5_ADMISSION.json")
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(v.gpu),MTO_PHYSICAL_GPU=str(v.gpu),
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",PYTHONWARNINGS="ignore",PYTHONUTF8="1")
    log=open(out/"train.log","a",buffering=1)
    cmd=["/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python",
        str(ROOT/"train.py"),v.arm]
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
    ticks=int(Path(f"/proc/{child.pid}/stat").read_text().split()[21])
    receipt={"classification":"Approved scratch native-vs-MTO paired100 one arm",
        "arm":v.arm,"pid":child.pid,"start_ticks":ticks,"cwd":str(ROOT),"cmd":cmd,
        "gpu":v.gpu,"gpu_uuid":before["uuid"],"health_before":before,
        "source_hashes":source,"code_hashes":code,
        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),
        "review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),
        "gpu5_admission_sha256":sha(ROOT/"GPU5_ADMISSION.json") if v.gpu==5 else None,
        "gpu5_admission_review_sha256":sha(ROOT/"GPU5_ADMISSION_REVIEW.json") if v.gpu==5 else None,
        "started":time.time(),"log":str(out/"train.log")}
    save_json(receipt,out/"LAUNCH_RECEIPT.json")
    time.sleep(3)
    assert child.poll() is None,f"Worker exited immediately with {child.returncode}"
    print(json.dumps({"state":"RUNNING","arm":v.arm,"pid":child.pid,
        "start_ticks":ticks,"gpu":v.gpu,"receipt":str(out/"LAUNCH_RECEIPT.json")}),flush=True)
if __name__=="__main__":main()

