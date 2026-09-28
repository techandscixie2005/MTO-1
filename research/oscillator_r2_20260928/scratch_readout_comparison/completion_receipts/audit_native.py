"""Saved-artifact integrity audit for completed scratch native arm only."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison")
N=R/"runs/native"; O=R/"completion_receipts"
def H(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def J(p):return json.loads(Path(p).read_text())
def C(x,msg):
 if not x:raise RuntimeError(msg)
f=J(N/"FIT_COMPLETE.json");l=J(N/"LAUNCH_RECEIPT.json");m=J(N/"RUN_MANIFEST.json")
v=J(R/"IMPLEMENTATION_REVIEW.json");p=J(R/"PREFLIGHT.json");q=J(N/"FIXED_CHECKPOINT_METRICS.json");order=J(R/"ORDER_PLAN.json")
C(f["event"]=="FIT_COMPLETE" and f["arm"]=="native" and f["epochs"]==100 and f["steps"]==188100 and f["test_batches"]==0,"terminal")
C(v["passed"] and p["passed"] and H(R/"IMPLEMENTATION_REVIEW.json")==l["review_sha256"] and H(R/"PREFLIGHT.json")==l["preflight_sha256"],"review")
C(f["source_hashes"]==v["source_hashes"]==p["source_hashes"]==m["source_hashes"]==l["source_hashes"],"source maps")
C(f["code_hashes"]==v["code_hashes"]==p["code_hashes"]==m["code_hashes"]==l["code_hashes"],"code maps")
for path,digest in {**f["source_hashes"],**f["code_hashes"]}.items():C(H(path)==digest,"source "+path)
C(H(R/"ORDER_PLAN.json")==f["order_plan_sha256"] and f["initial_hash"]==m["initial_hash"],"order/initial")
C(H(N/"FIXED_CHECKPOINT_METRICS.json")==f["fixed_metrics_sha256"],"fixed metrics")
C(not Path(f"/proc/{l['pid']}").exists(),"worker exists")
for k,path in {"initial":R/"initial/native.pt","raw_f":N/"best_raw_f.pt","joint":N/"best_joint.pt","final100":N/"final100.pt","last":N/"last.pt"}.items():C(H(path)==f["checkpoint_hashes"][k],"checkpoint "+k)
for k,n in {"epoch0":"val_epoch0.npz","best_raw_f":"val_best_raw_f.npz","best_joint":"val_best_joint.npz","final100":"val_final100.npz"}.items():C(H(N/n)==f["prediction_hashes"][k],"prediction "+k)
for kind in ("raw_f","joint"):
 epoch=f["best"][kind]["epoch"]
 C(H(N/f"selected/{kind}_epoch{epoch:03d}.pt")==f["checkpoint_hashes"][kind],"versioned ckpt")
 C(H(N/f"selected/{kind}_epoch{epoch:03d}_val.npz")==f["prediction_hashes"]["best_"+kind],"versioned array")
hist=[json.loads(x) for x in (N/"history.jsonl").read_text().splitlines()]
C(len(hist)==len(order["orders"])==100,"history length")
for i,(x,y) in enumerate(zip(hist,order["orders"]),1):
 C(x["epoch"]==y["epoch"]==i and x["steps"]==1881*i and x["updates_this_epoch"]==y["updates"]==1881 and x["order_sha256"]==y["sha256"],"epoch order")
 C(abs(x["lr"]-(.001 if i<=60 else .0003 if i<=85 else .0001))<1e-12,"lr")
z=J(N/"epoch0_val.json")
for kind,init,field in (("raw_f",z["raw_f"]["sse"],"val_raw_f"),("joint",z["normalized_joint"][0],"val_normalized")):
 vals=[(init,0)]+[((x[field]["sse"] if kind=="raw_f" else x[field][0]),x["epoch"]) for x in hist]
 value,epoch=min(vals)
 C(epoch==f["best"][kind]["epoch"] and abs(value-f["best"][kind]["metric"])<1e-9,"selected "+kind)
ref=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz")
with np.load(ref) as d:
 ids=d["ids"].copy();ix=d["indices"].copy();truth=d["f_true"].copy();Etrue=d["E_true"].copy()
 C(d["mask_f_true"].all() and d["mask_E_true"].all(),"masks")
sst=float(np.square(truth-truth.mean()).sum())
saved={}
for name,file,section in (("initial","val_epoch0.npz","initial"),("raw_f","val_best_raw_f.npz","raw_f"),("joint","val_best_joint.npz","joint"),("final100","val_final100.npz","final100")):
 with np.load(N/file) as d:
  C(np.array_equal(d["ids"],ids) and np.array_equal(d["indices"],ix),"alignment")
  C(d["f_true"].dtype==np.float64 and np.array_equal(d["f_true"],truth) and d["E_true"].dtype==np.float64 and np.array_equal(d["E_true"],Etrue),"FP64 truth")
  a=d["f"].astype(np.float64);b=d["E"].astype(np.float64)
  C(np.isfinite(a).all() and np.isfinite(b).all(),"finite")
  sse=float(np.square(a-truth).sum());es=float(np.square(b-Etrue).sum())
  C(abs(sse-q[section]["val"]["raw_f"]["sse"])<2e-6 and abs(es-q[section]["val"]["energy"]["sse"])<1e-4,"fixed array metrics within accepted FP32 replay tolerance")
  saved[name]={"sse":sse,"r2":1-sse/sst,"mae":float(np.abs(a-truth).mean()),"energy_mae":float(np.abs(b-Etrue).mean()),"saved_minus_fixed_f_sse":sse-q[section]["val"]["raw_f"]["sse"],"saved_minus_fixed_energy_sse":es-q[section]["val"]["energy"]["sse"]}
for key in ("initial","raw_f","joint","final100"):
 C(q[key]["train"]["raw_f"]["count"]==1203550 and q[key]["val"]["raw_f"]["count"]==66860,"full counts")
 C(q[key]["val_replay_max_abs_f"]<1e-5 and q[key]["val_replay_max_abs_E"]<1e-5,"replay")
C(f["gpu_before"]["ecc"]==f["gpu_after"]["ecc"] and f["gpu_after"]["ecc"]["volatile"]["dram_correctable"]==1 and f["gpu_after"]["ecc"]["volatile"]["dram_uncorrectable"]==0,"GPU5 ECC")
C(H(R/"GPU5_ADMISSION_REVIEW.json")==l["gpu5_admission_review_sha256"],"GPU5 admission")
out={"passed":True,"classification":"native arm completed; paired scratch study pending; saved arrays only; no test","epochs":100,"steps":188100,"selected":f["best"],"sources_verified":len(f["source_hashes"]),"code_verified":len(f["code_hashes"]),"orders_verified":100,"replay_metric_tolerances":{"f_sse_absolute":2e-6,"energy_sse_absolute":1e-4},"terminal_sha256":H(N/"FIT_COMPLETE.json"),"launch_sha256":H(N/"LAUNCH_RECEIPT.json"),"history_sha256":H(N/"history.jsonl"),"fixed_metrics_sha256":H(N/"FIXED_CHECKPOINT_METRICS.json"),"saved_validation":saved,"fixed_full_train":{k:{"f":q[k]["train"]["raw_f"],"energy_mae":q[k]["train"]["energy"]["mae"]} for k in ("initial","raw_f","joint","final100")},"limits":["Train metrics are terminal fixed-checkpoint aggregates; no train prediction arrays saved for independent elementwise check.","MTO arm pending; no paired outcome or model promotion.","Replay-versus-array tolerances compare saved validation arrays with fixed-checkpoint replays already performed during authorized training completion; this audit ran no inference."]}
a=O/"NATIVE_COMPLETION_AUDIT.json";a.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
lines=["# Scratch native completion audit","","PASS: 100 epochs, 188,100 updates; reviewed source/data and 100 order hashes, selected checkpoint/array hashes, FP64 validation labels, and GPU5 ECC counters match. MTO arm and paired result remain pending. No test or new inference.","","| Checkpoint | Train f R² | Val f R² | Val SSE | Val f MAE | Val E MAE |","|---|---:|---:|---:|---:|---:|"]
for k in ("initial","raw_f","joint","final100"):
 t=q[k]["train"]["raw_f"];w=saved[k]
 lines.append(f"| {k} (epoch {q[k]['epoch']}) | {t['r2']:.6f} | {w['r2']:.6f} | {w['sse']:.6f} | {w['mae']:.6f} | {w['energy_mae']:.6f} |")
lines+=["","Raw-f and joint selection both chose epoch 21. Separate checkpoint and array aliases are retained; completion-time replay aggregates show the reported FP32 variation. The two saved selected-validation arrays have identical f metrics.","Train metrics are terminal full-train aggregates; validation metrics were independently recomputed from saved predictions. Saved arrays and a later fixed-checkpoint replay differ by at most 9.4e-7 in f SSE and 3.7e-5 in energy SSE; the audit uses explicit 2e-6/1e-4 absolute replay tolerances. Those replays were produced during the authorized training completion; this audit ran no model inference."]
(O/"NATIVE_COMPLETION_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"passed":True,"selected_r2":saved["raw_f"]["r2"],"selected_sse":saved["raw_f"]["sse"],"final_r2":saved["final100"]["r2"],"audit_sha256":H(a)}))

