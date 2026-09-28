"""Saved-artifact audit of completed seed37 only; never loads a model or test rows."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
R=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928/eta0_seed_replication")
N=R/"runs/seed37";O=R/"completion_receipts"
def H(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def C(x,msg):
 if not x:raise RuntimeError(msg)
f=J(N/"FIT_COMPLETE.json");l=J(N/"LAUNCH_RECEIPT.json");m=J(N/"RUN_MANIFEST.json");v=J(R/"IMPLEMENTATION_REVIEW.json");p=J(R/"PREFLIGHT.json");q=J(N/"FIXED_CHECKPOINT_METRICS.json");order=J(R/"ORDER_PLAN.json")
C(f["event"]=="FIT_COMPLETE" and f["seed"]==37 and f["epochs"]==100 and f["steps"]==188100 and f["test_batches"]==0,"terminal")
C(v["passed"] and p["passed"] and H(R/"IMPLEMENTATION_REVIEW.json")==l["review_sha256"] and H(R/"PREFLIGHT.json")==l["preflight_sha256"],"review")
C(f["source_hashes"]==v["source_hashes"]==p["source_hashes"]==m["source_hashes"]==l["source_hashes"],"source maps")
C(f["code_hashes"]==v["code_hashes"]==p["code_hashes"]==m["code_hashes"]==l["code_hashes"],"code maps")
for path,digest in {**f["source_hashes"],**f["code_hashes"]}.items():C(H(path)==digest,"source "+path)
C(H(R/"ORDER_PLAN.json")==f["order_hashes_sha256"] and f["initial_hash"]==m["initial_state_sha256"],"order/initial")
C(H(N/"FIXED_CHECKPOINT_METRICS.json")==f["fixed_metrics_sha256"],"fixed metrics")
C(not Path("/proc/"+str(l["pid"])).exists(),"worker exists")
for k,path in {"initial":N/"initial.pt","legacy":N/"best_legacy.pt","raw_f":N/"best_raw_f.pt","final100":N/"final100.pt","last":N/"last.pt"}.items():C(H(path)==f["checkpoint_hashes"][k],"checkpoint "+k)
for k,path in {"epoch0":N/"val_epoch0.npz","best_legacy":N/"val_best_legacy.npz","best_raw_f":N/"val_best_raw_f.npz","final100":N/"val_final100.npz"}.items():C(H(path)==f["val_prediction_hashes"][k],"prediction "+k)
for kind in ("legacy","raw_f"):
 epoch=f["best_"+kind+"_epoch"]
 C(H(N/f"selected/{kind}_epoch{epoch:03d}.pt")==f["checkpoint_hashes"][kind],"selected ckpt")
 C(H(N/f"selected/{kind}_epoch{epoch:03d}_val.npz")==f["val_prediction_hashes"]["best_"+kind],"selected array")
hist=[json.loads(x) for x in (N/"history.jsonl").read_text().splitlines()]
C(len(hist)==len(order["seeds"]["37"])==100,"history length")
for i,(x,y) in enumerate(zip(hist,order["seeds"]["37"]),1):
 C(x["epoch"]==y["epoch"]==i and x["steps"]==1881*i and x["updates_this_epoch"]==1881 and x["order_sha256"]==y["order_sha256"],"epoch order")
z=J(N/"epoch0_val.json")
C(min([(z["legacy_val_objective"][0],0)]+[(x["val_legacy"][0],x["epoch"]) for x in hist])[1]==f["best_legacy_epoch"],"legacy selection")
C(min([(z["raw_native_f"]["sse"],0)]+[(x["val_raw_f"]["sse"],x["epoch"]) for x in hist])[1]==f["best_raw_f_epoch"],"raw selection")
C(abs(hist[f["best_legacy_epoch"]-1]["val_legacy"][0]-f["best_legacy"])<1e-10 and abs(hist[f["best_legacy_epoch"]-1]["val_raw_f"]["sse"]-f["best_raw_f_sse"])<1e-8,"selection values")
ref=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz")
with np.load(ref) as d:
 ids=d["ids"].copy();ix=d["indices"].copy();truth=d["f_true"].copy();Etrue=d["E_true"].copy()
 C(d["mask_f_true"].all() and d["mask_E_true"].all(),"masks")
sst=float(np.square(truth-truth.mean()).sum());saved={}
for name,file,section in (("initial","val_epoch0.npz","initial"),("legacy","val_best_legacy.npz","legacy"),("raw_f","val_best_raw_f.npz","raw_f"),("final100","val_final100.npz","final100")):
 with np.load(N/file) as d:
  C(np.array_equal(d["ids"],ids) and np.array_equal(d["indices"],ix),"alignment")
  C(d["f_true"].dtype==np.float64 and np.array_equal(d["f_true"],truth) and d["E_true"].dtype==np.float64 and np.array_equal(d["E_true"],Etrue),"FP64 truth")
  a=d["f"].astype(np.float64);b=d["E"].astype(np.float64)
  C(np.isfinite(a).all() and np.isfinite(b).all(),"finite")
  sse=float(np.square(a-truth).sum());es=float(np.square(b-Etrue).sum())
  C(abs(sse-q[section]["val"]["raw_native_f"]["sse"])<2e-5 and abs(es-q[section]["val"]["energy"]["sse"])<1e-3,"completion replay tolerance")
  saved[name]={"sse":sse,"r2":1-sse/sst,"mae":float(np.abs(a-truth).mean()),"energy_mae":float(np.abs(b-Etrue).mean()),"saved_minus_completion_replay_f_sse":sse-q[section]["val"]["raw_native_f"]["sse"]}
for key in ("initial","legacy","raw_f","final100"):
 C(q[key]["train"]["raw_native_f"]["count"]==1203550 and q[key]["val"]["raw_native_f"]["count"]==66860,"full counts")
 C(q[key]["val_prediction_replay_max_abs_f"]<2e-5 and q[key]["val_prediction_replay_max_abs_E"]<1e-5,"replay")
C(f["gpu_before"]["uuid"]==f["gpu_after"]["uuid"]==l["gpu_uuid"] and f["gpu_before"]["ecc"]==f["gpu_after"]["ecc"],"GPU1 health")
C(f["val_prediction_hashes"]["best_legacy"]==f["val_prediction_hashes"]["best_raw_f"],"selected arrays equal")
ckl=torch.load(N/"best_legacy.pt",map_location="cpu",weights_only=False);ckr=torch.load(N/"best_raw_f.pt",map_location="cpu",weights_only=False)
C(ckl["epoch"]==ckr["epoch"]==f["best_legacy_epoch"]==f["best_raw_f_epoch"],"same epoch")
C(ckl["model"].keys()==ckr["model"].keys() and all(torch.equal(ckl["model"][k],ckr["model"][k]) for k in ckl["model"]),"model tensors differ")
C(all(ckl[k]==ckr[k] for k in ("config","source_hashes","code_hashes")),"checkpoint metadata source")
C(ckl["selection"]=="legacy" and ckr["selection"]=="raw_f_secondary" and ckl["metric"]==f["best_legacy"] and ckr["metric"]==f["best_raw_f_sse"],"selection metadata")
out={"passed":True,"classification":"seed37 individual complete; family summary audited separately; no test or audit-time inference","epochs":100,"steps":188100,"selected":{"legacy":{"epoch":f["best_legacy_epoch"],"metric":f["best_legacy"]},"raw_f":{"epoch":f["best_raw_f_epoch"],"sse":f["best_raw_f_sse"]}},"sources_verified":len(f["source_hashes"]),"code_verified":len(f["code_hashes"]),"orders_verified":100,"selected_model_tensors_bitwise_identical":len(ckl["model"]),"selected_checkpoint_metadata_difference":{"legacy":{"selection":ckl["selection"],"metric":ckl["metric"]},"raw_f":{"selection":ckr["selection"],"metric":ckr["metric"]}},"terminal_sha256":H(N/"FIT_COMPLETE.json"),"launch_sha256":H(N/"LAUNCH_RECEIPT.json"),"history_sha256":H(N/"history.jsonl"),"fixed_metrics_sha256":H(N/"FIXED_CHECKPOINT_METRICS.json"),"saved_validation":saved,"fixed_full_train":{k:{"f":q[k]["train"]["raw_native_f"],"energy_mae":q[k]["train"]["energy"]["mae"]} for k in ("initial","legacy","raw_f","final100")},"limits":["Full-train figures are completion-time aggregates, not independently recomputed from saved train predictions.","The selected checkpoint model tensors/buffers are bitwise identical; serialized checkpoint hashes differ only because metric and selection metadata differ. Selected validation array bytes are identical.","Replay comparisons use already-recorded completion-time replays; this audit ran no inference.","The prespecified three-seed ensemble is reported separately in the completed family summary."]}
a=O/"SEED37_COMPLETION_AUDIT.json";a.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
lines=["# Seed37 individual completion audit","","PASS: 100 epochs and 188,100 updates. Reviewed source and 100 order hashes, selected checkpoint/array hashes, raw FP64 validation labels, and GPU1 ECC counters match. The three-seed family result is reported separately. No test access or audit-time inference.","","| Checkpoint | Train f R² | Val f R² | Val SSE | Val f MAE | Val E MAE |","|---|---:|---:|---:|---:|---:|"]
for k in ("initial","legacy","raw_f","final100"):
 t=q[k]["train"]["raw_native_f"];w=saved[k]
 lines.append(f"| {k} (epoch {q[k][chr(101)+chr(112)+chr(111)+chr(99)+chr(104)]}) | {t[chr(114)+chr(50)]:.6f} | {w[chr(114)+chr(50)]:.6f} | {w[chr(115)+chr(115)+chr(101)]:.6f} | {w[chr(109)+chr(97)+chr(101)]:.6f} | {w[chr(101)+chr(110)+chr(101)+chr(114)+chr(103)+chr(121)+chr(95)+chr(109)+chr(97)+chr(101)]:.6f} |")
lines += ["","Legacy and raw-f selection both chose epoch53; their saved validation arrays have the same SHA-256. Train metrics are completion-time full-train aggregates; saved validation metrics are recomputed here. Saved-array versus completion-time replay differences are recorded within the declared FP32 tolerances.","","The primary family comparison uses legacy-selected seeds after both complete. Raw-f-selected and mixed-selection outputs are separately labeled secondary analyses. No family comparison or promotion is made here."]
(O/"SEED37_COMPLETION_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"passed":True,"selected_r2":saved["legacy"]["r2"],"selected_sse":saved["legacy"]["sse"],"final_r2":saved["final100"]["r2"],"audit_sha256":H(a)}))
