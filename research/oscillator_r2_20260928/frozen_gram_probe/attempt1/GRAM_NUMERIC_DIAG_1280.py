#!/usr/bin/env python3
"""Bounded first1280 TRAIN molecule cache/repeatability diagnostic after stopped Gram run."""
import fcntl,json,os,sys,time
import numpy as np,torch
from features import ROOT,ARCH,Split,source_model,setup_torch,load_existing_cache,sha,live_features,save_json
gpu=int(os.environ["MTO_PHYSICAL_GPU"])
assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and gpu in (1,6)
lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(ARCH));import gpu_health
before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
setup_torch(11)
split=Split("train");cache=load_existing_cache("train");model=source_model("cuda").eval()
n=1280
summary={"classification":"Fixed first1280 TRAIN molecule source/cache numerical diagnosis only; no fit/validation",
         "gpu":gpu,"time":time.time(),"rows":n,
         "cache_sha256":sha(ROOT.parent/"frozen_residual/runtime/cache_train.npz"),"keys":{}}
for k in ("h","E_pred","base32","base64"):
 summary["keys"][k]={"max_cache_abs":0.,"max_repeat_abs":0.,
                     "cache_old_tolerance_fail_count":0,
                     "max_cache_error_over_old_tolerance":0.,
                     "first_failure":None}
for a in range(0,n,64):
 b=a+64;x=split.batch(a,b,"cuda")
 one=live_features(model,x);two=live_features(model,x)
 for k,s in summary["keys"].items():
  v=one[k].cpu().numpy();w=two[k].cpu().numpy();ref=cache[k][a:b]
  err=np.abs(v-ref);tol=1e-6+1e-5*np.abs(ref)
  bad=err>tol
  s["max_cache_abs"]=max(s["max_cache_abs"],float(err.max()))
  s["max_repeat_abs"]=max(s["max_repeat_abs"],float(np.abs(v-w).max()))
  s["cache_old_tolerance_fail_count"]+=int(bad.sum())
  s["max_cache_error_over_old_tolerance"]=max(s["max_cache_error_over_old_tolerance"],float((err/tol).max()))
  if bad.any() and s["first_failure"] is None:
   j=np.argwhere(bad)[0]
   s["first_failure"]={"batch_start":a,"local_index":j.tolist(),
                       "cached":float(ref[tuple(j)]),"live":float(v[tuple(j)]),
                       "abs_error":float(err[tuple(j)]),"old_tolerance":float(tol[tuple(j)])}
after=gpu_health.snapshot(gpu)
assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
summary["health_before"]=before;summary["health_after"]=after
save_json(summary,ROOT/"GRAM_NUMERIC_DIAG_1280.json")
print(json.dumps({k:{"max_cache":v["max_cache_abs"],"max_repeat":v["max_repeat_abs"],
   "bad":v["cache_old_tolerance_fail_count"],"first":v["first_failure"]}
   for k,v in summary["keys"].items()}))

