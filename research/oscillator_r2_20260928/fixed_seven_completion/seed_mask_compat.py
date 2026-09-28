"""Reviewed adapter for seed NPZ files that omit an all-valid f mask."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928")
HERE=ROOT/"fixed_seven_completion"
OLD=ROOT/"fixed_seven_sidecar"
sys.path.insert(0,str(OLD))
import fixed_seven as fs
SEED_PATHS={ROOT/f"eta0_seed_replication/runs/seed{s}/val_best_legacy.npz" for s in (23,37)}
ORIGINAL=fs.load_strict

def H(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def check(ok,msg):
 if not ok:raise RuntimeError(msg)
def load_seed(path,rows,expected):
 path=Path(path)
 check(path.is_file() and H(path)==expected,"seed prediction hash/missing")
 mask=np.asarray(rows["mask"])
 check(mask.dtype==np.bool_ and mask.shape==(len(rows["ids"]),10) and mask.all(),"authoritative seed mask not all-valid")
 with np.load(path,allow_pickle=False) as z:
  check(set(("indices","ids","f_true","E_true","f")).issubset(z.files),"seed array keys")
  ix=z["indices"];ids=z["ids"];truth=z["f_true"];Etruth=z["E_true"];pred=z["f"]
  if "mask_f_true" in z.files:check(z["mask_f_true"].dtype==np.bool_ and np.array_equal(z["mask_f_true"],mask),"optional seed mask mismatch")
 check(np.array_equal(ix,rows["indices"]) and np.array_equal(ids,rows["ids"]),"seed IDs/indices mismatch")
 check(truth.dtype==np.float64 and np.array_equal(truth,rows["truth"]),"seed FP64 f truth mismatch")
 check(Etruth.dtype==np.float64 and np.array_equal(Etruth,rows["E_true"]),"seed FP64 E truth mismatch")
 check(pred.shape==truth.shape and np.issubdtype(pred.dtype,np.floating) and np.isfinite(pred).all(),"seed prediction invalid")
 return np.asarray(pred,dtype=np.float64)
def dispatch(path,rows,expected):
 if Path(path).resolve() in SEED_PATHS:return load_seed(path,rows,expected)
 return ORIGINAL(path,rows,expected)
def reviewed_gate():
 fs.reviewed_execution_gate()
 p=HERE/"COMPATIBILITY_REVIEW.json"
 check(p.is_file(),"PENDING: compatibility independent review")
 r=json.loads(p.read_text(encoding="utf-8"))
 check(r.get("passed") is True and r.get("adapter_sha256")==H(__file__) and
       r.get("protocol_sha256")==H(HERE/"COMPATIBILITY_PROTOCOL.md") and
       r.get("preflight_sha256")==H(HERE/"COMPATIBILITY_PREFLIGHT.json") and
       r.get("original_script_sha256")==H(OLD/"fixed_seven.py") and
       r.get("original_review_sha256")==H(OLD/"IMPLEMENTATION_REVIEW.json"),"compatibility exact-hash review mismatch")
 return r

def main():
 reviewed_gate()
 out=OLD/"SEED_FIXED7_RESULTS.json"
 check(not out.exists() and not (OLD/"SEED_FIXED7_REPORT.md").exists(),"completed output already exists")
 # Frozen comparator reference supplies the all-valid mask and raw FP64 energy truth.
 ref=ROOT/"../.."/"experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz"
 with np.load(ref,allow_pickle=False) as z:
  mask=z["mask_f_true"].copy();Etrue=z["E_true"].copy();ix=z["indices"].copy();ids=z["ids"].copy()
 check(mask.dtype==np.bool_ and mask.shape==(6686,10) and mask.all(),"reference mask")
 check(Etrue.dtype==np.float64 and Etrue.shape==(6686,10),"reference energy truth")
 old_loader=fs.load_strict
 def adapter(path,rows,expected):
  if Path(path).resolve() in SEED_PATHS:
   check(np.array_equal(rows["indices"],ix) and np.array_equal(rows["ids"],ids),"reference alignment")
   return load_seed(path,{**rows,"E_true":Etrue},expected)
  return old_loader(path,rows,expected)
 fs.load_strict=adapter
 try:fs.completed_seed()
 finally:fs.load_strict=old_loader
 check(out.exists() and (OLD/"SEED_FIXED7_REPORT.md").exists(),"completed output missing")
 receipt={"classification":"Reviewed seed mask compatibility for unchanged fixed-seven arithmetic","time":time.time(),"adapter_sha256":H(__file__),"protocol_sha256":H(HERE/"COMPATIBILITY_PROTOCOL.md"),"compatibility_review_sha256":H(HERE/"COMPATIBILITY_REVIEW.json"),"original_script_sha256":H(OLD/"fixed_seven.py"),"original_review_sha256":H(OLD/"IMPLEMENTATION_REVIEW.json"),"frozen_mask_source_sha256":H(ref),"result_sha256":H(out),"report_sha256":H(OLD/"SEED_FIXED7_REPORT.md"),"test_access":False,"inference":False}
 (HERE/"SEED_COMPAT_EXECUTION.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
if __name__=="__main__":main()
