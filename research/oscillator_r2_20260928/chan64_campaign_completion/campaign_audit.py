#!/usr/bin/env python3
"""Read-only integrity audit of the completed legacy chan64 campaign."""
import hashlib, json, math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path("/home/inspur/MTO-1")
SRC=ROOT/"experiments/qm9s_chan64_20260928"
OUT=ROOT/"research/oscillator_r2_20260928/chan64_campaign_completion"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def local(t):return datetime.fromtimestamp(t,ZoneInfo("Asia/Shanghai")).isoformat()
prior=json.loads((ROOT/"research/oscillator_r2_20260928/chan64_validation_only.json").read_text())
data_hash=json.loads((SRC/"data/hashes.json").read_text())
for name in ("dataset.npz","raw_labels.npz","splits.json","normalization.json","trace_weight.json"):
 assert sha(SRC/"data"/name)==data_hash[name],name
runs={}
for name in ("G1","G2","G3","G4"):
 rd=SRC/"runs"/name
 man=json.loads((rd/"run_manifest.json").read_text())
 cfg=json.loads((SRC/"configs"/(name+".json")).read_text())
 hist=[json.loads(x) for x in (rd/"history.jsonl").read_text().splitlines()]
 assert cfg==man["config"]
 assert [r["epoch"] for r in hist]==list(range(1,len(hist)+1))
 for rel,digest in man["fingerprint"].items():
  assert sha(SRC/rel)==digest,(name,rel)
 p=prior["results"][name]
 assert sha(rd/"best.pt")==p["checkpoint_sha256"]
 assert hist[p["checkpoint_epoch"]-1]["val"][0]==p["checkpoint_selection_val"]
 assert p["checkpoint_selection_val"]==min(r["val"][0] for r in hist)
 if name=="G2":
  terminal=json.loads((rd/"ADMIN_STOPPED.json").read_text())
  assert terminal["event"]=="ADMIN_STOPPED" and terminal["g2_fit_complete_absent"]
  assert not (rd/"FIT_COMPLETE.json").exists()
  assert terminal["after"]["best_pt_sha256"]==p["checkpoint_sha256"]
  classification="administratively_stopped_incomplete"
  actual_epochs=len(hist)
 else:
  terminal=json.loads((rd/"FIT_COMPLETE.json").read_text())
  assert terminal["event"]=="FIT_COMPLETE" and terminal["epochs"]==len(hist)
  assert terminal["best_epoch"]==p["checkpoint_epoch"]
  assert terminal["best_val"]==p["checkpoint_selection_val"]
  assert terminal["reason"]=="early_stop"
  classification="natural_early_stop"
  actual_epochs=terminal["epochs"]
 runs[name]={"classification":classification,"epochs_completed":actual_epochs,
  "best_epoch":p["checkpoint_epoch"],"selection_metric":"validation training objective, not oscillator-strength R2",
  "selected_joint_loss":p["checkpoint_selection_val"],
  "f_evaluation_mode":p["f_mode"],"pooled_raw_f_validation":{"r2":p["val_f_r2"],"sse":p["val_f_sse"],"rmse":p["val_f_rmse"]},
  "selected_energy_loss_normalized":hist[p["checkpoint_epoch"]-1]["val"][1] if cfg["eta"]==1 else None,
  "selected_trace_loss_weighted_normalized":hist[p["checkpoint_epoch"]-1]["val"][2],
  "selected_traceless_tensor_loss_normalized":hist[p["checkpoint_epoch"]-1]["val"][3] if cfg["eta"]==1 else None,
  "terminal_time_local":local(terminal["time"]),
  "artifact_sha256":{f:sha(rd/f) for f in ("run_manifest.json","history.jsonl","best.pt","last.pt",
                                         "ADMIN_STOPPED.json" if name=="G2" else "FIT_COMPLETE.json")},
  "source_fingerprint_sha256":sha(rd/"run_manifest.json"),
  "terminal":terminal}
sse0=94.20512092938401
sse3=87.2150742627189
r20=.4052941183410983
r23=.4494214632744171
g3=runs["G3"]["pooled_raw_f_validation"]
report={"classification":"completed legacy chan64 campaign with one intentional incomplete arm",
 "no_new_inference":True,"no_test_access":True,
 "validation_audit_source_sha256":sha(ROOT/"research/oscillator_r2_20260928/chan64_validation_only.json"),
 "validation_evaluator_sha256":sha(ROOT/"research/oscillator_r2_20260928/chan64_validation_only.py"),
 "data_hashes_actual_verified":{x:sha(SRC/"data"/x) for x in ("dataset.npz","raw_labels.npz","splits.json","normalization.json","trace_weight.json")},
 "split_counts":{"train":120355,"validation":6686,"test":6686},
 "label_source":"raw printed FP64 oscillator strengths; all validation masks valid; 10 states per molecule",
 "checkpoint_selection":"minimum legacy validation objective; native raw-f R2 separately evaluated after selection",
 "source_and_scaler":"each run_manifest config/fingerprint matches current frozen source; normalization and E2 trace weight train-only",
 "campaign_terminal_status":json.loads((SRC/"reports/CAMPAIGN_BLOCKED.json").read_text()),
 "supervisor_status_sha256":sha(SRC/"supervisor_status.json"),
 "runs":runs,
 "context":{"eta0":{"r2":r20,"sse":sse0},"fixed_equal3":{"r2":r23,"sse":sse3},
  "G3_delta_r2_vs_eta0":g3["r2"]-r20,"G3_delta_sse_vs_eta0":g3["sse"]-sse0,
  "G3_delta_r2_vs_equal3":g3["r2"]-r23,"G3_delta_sse_vs_equal3":g3["sse"]-sse3},
 "limitations":["G2 and G4 use true E for their historical f diagnostic because their E heads were not trained; their f numbers are not deployable native scores.",
                "Historical validation already selected checkpoints; no independent confirmatory holdout is claimed.",
                "The old validation audit saved aggregate scores only; no G1/G3 per-molecule predictions for ensemble assessment.",
                "Raw-f and energy MAE are unavailable from prior aggregate scores; a separate authorized selected-validation replay is required."]}
p=OUT/"CAMPAIGN_AUDIT.json"
assert not p.exists()
p.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"path":str(p),"sha256":sha(p),"G3":g3,"campaign":{k:runs[k]["classification"] for k in runs}},indent=2))
