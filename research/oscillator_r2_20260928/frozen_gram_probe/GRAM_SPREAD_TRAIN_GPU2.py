#!/usr/bin/env python3
"""Prespecified spread-train numerical/device check; no fit or validation."""
import fcntl,json,os,sys,time
import numpy as np,torch
from features import ROOT,ARCH,Split,source_model,setup_torch,load_existing_cache,sha,live_features,save_json
gpu=int(os.environ["MTO_PHYSICAL_GPU"])
assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and gpu==2
lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(ARCH));import gpu_health
before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
setup_torch(11)
split=Split("train");cache=load_existing_cache("train");model=source_model("cuda").eval()
last_aligned=((len(split.indices)-64)//64)*64
starts=np.rint(np.linspace(0,last_aligned,16)/64).astype(int)*64
starts=np.unique(np.r_[starts,len(split.indices)-35]).tolist()
result={"classification":"Prespecified16 evenly spaced full64 TRAIN batches plus final35, two frozen forwards each on clean GPU2; no fit/val/test",
        "created":time.time(),"gpu":gpu,"starts":starts,
        "cache_sha256":sha(ROOT.parent/"frozen_residual/runtime/cache_train.npz"),
        "source_model_checkpoint_sha256":sha(ROOT.parent/"architecture/initial/retained_residual.pt"),
        "health_before":before,"keys":{},"gram_repeat_max_abs":0.,"trace_mismatch_count":0}
for key in ("h","E_pred","base32","base64"):
 result["keys"][key]={"atol":5e-6 if key=="h" else 1e-6,"rtol":1e-5,
        "max_cache_abs":0.,"max_live_repeat_abs":0.,"failure_count":0}
for a in starts:
 b=min(a+64,len(split.indices))
 x=split.batch(a,b,"cuda")
 first=live_features(model,x);second=live_features(model,x)
 result["gram_repeat_max_abs"]=max(result["gram_repeat_max_abs"],
    float((first["gram"]-second["gram"]).abs().max()))
 for live in (first,second):
  if not torch.allclose(live["trace_gram"],live["trace_source"],rtol=1e-5,atol=2e-5):
   result["trace_mismatch_count"]+=1
 for key,s in result["keys"].items():
  ref=cache[key][a:b]
  v=first[key].cpu().numpy();w=second[key].cpu().numpy()
  for q in (v,w):
   err=np.abs(q-ref)
   tol=s["atol"]+s["rtol"]*np.abs(ref)
   s["max_cache_abs"]=max(s["max_cache_abs"],float(err.max()))
   s["failure_count"]+=int((err>tol).sum())
  s["max_live_repeat_abs"]=max(s["max_live_repeat_abs"],float(np.abs(v-w).max()))
assert result["trace_mismatch_count"]==0
assert all(v["failure_count"]==0 for v in result["keys"].values())
after=gpu_health.snapshot(gpu)
assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
result["health_after"]=after;result["passed"]=True
save_json(result,ROOT/"GRAM_SPREAD_TRAIN_GPU2.json")
print(json.dumps({"passed":True,"starts":len(starts),"gram_repeat_max":result["gram_repeat_max_abs"],
    "keys":result["keys"]}))

