#!/usr/bin/env python3
"""Bounded fixed first64 TRAIN molecule cache and repeatability diagnosis."""
import fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from features import ROOT,ARCH,Split,source_model,setup_torch,load_existing_cache,sha,live_features,save_json
gpu=int(os.environ["MTO_PHYSICAL_GPU"])
assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and gpu in (1,6)
lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(ARCH));import gpu_health
before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
setup_torch(11)
split=Split("train");cache=load_existing_cache("train")
model=source_model("cuda").eval()
x=split.batch(0,64,"cuda")
runs=[]
for t in range(8):
    live=live_features(model,x)
    runs.append({k:live[k].cpu().numpy() for k in ("h","E_pred","base32","base64")})
out={"classification":"Fixed first64 train molecules only, eight repeated source forwards; no fit/validation",
     "gpu":gpu,"time":time.time(),"cache_sha256":sha(ROOT.parent/"frozen_residual/runtime/cache_train.npz"),
     "keys":{}}
for k in ("h","E_pred","base32","base64"):
    ref=cache[k][:64]
    shape=ref.shape;dtype=str(ref.dtype)
    rows=[]
    for j,r in enumerate(runs):
        v=r[k];err=np.abs(v-ref)
        tol=1e-6+1e-5*np.abs(ref)
        rows.append({"repeat":j,"cache_max_abs":float(err.max()),
             "cache_p999_abs":float(np.quantile(err,.999)),
             "cache_allclose_failure_count":int((err>tol).sum()),
             "cache_max_error_over_old_tolerance":float((err/tol).max()),
             "first_repeat_max_abs":float(np.abs(v-runs[0][k]).max())})
    out["keys"][k]={"shape":shape,"dtype":dtype,"repeats":rows,
        "cache_magnitude_quantiles":np.quantile(np.abs(ref),[0,.5,.9,.99,1]).tolist()}
after=gpu_health.snapshot(gpu)
assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
out["health_before"]=before;out["health_after"]=after
save_json(out,ROOT/"GRAM_NUMERIC_DIAG.json")
print(json.dumps({k:{"max_cache":max(r["cache_max_abs"] for r in v["repeats"]),
    "max_repeat":max(r["first_repeat_max_abs"] for r in v["repeats"]),
    "failed_elements":max(r["cache_allclose_failure_count"] for r in v["repeats"])}
    for k,v in out["keys"].items()}))

