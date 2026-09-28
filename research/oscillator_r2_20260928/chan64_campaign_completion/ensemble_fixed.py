#!/usr/bin/env python3
"""Fixed exploratory validation ensembles from sealed saved predictions; no inference."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/home/inspur/MTO-1")
OUT=ROOT/"research/oscillator_r2_20260928/chan64_campaign_completion"
ETA=ROOT/"experiments/qm9s_eta_Ef_20260926/runs"
IDENTITY=ROOT/"experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:
  keys=("ids","indices","f","f_true","mask_f_true")
  assert all(k in z.files for k in keys),(p,z.files)
  return {k:np.asarray(z[k]).copy() for k in keys}
def metrics(y,p,m):
 e=p[m]-y[m];v=y[m];sse=float(np.square(e).sum(dtype=np.float64))
 sst=float(np.square(v-v.mean()).sum(dtype=np.float64))
 return {"count":int(len(v)),"sse":sse,"sst":sst,"r2":1-sse/sst,
         "rmse":float(np.sqrt(sse/len(v))),"mae":float(np.abs(e).mean())}
def group_sums(y,p0,p1,groups):
 n=[];sy=[];sy2=[];sse0=[];sse1=[]
 for rows in groups:
  yy=y[rows];a=p0[rows];b=p1[rows]
  n.append(yy.size);sy.append(yy.sum());sy2.append(np.square(yy).sum())
  sse0.append(np.square(a-yy).sum());sse1.append(np.square(b-yy).sum())
 return [np.asarray(x,dtype=np.float64) for x in (n,sy,sy2,sse0,sse1)]
def bootstrap(y,base,pred,units,seed=20260929,reps=2000):
 n,sy,sy2,s0,s1=group_sums(y,base,pred,units)
 rng=np.random.default_rng(seed);d=[]
 for _ in range(reps):
  draw=rng.integers(0,len(units),len(units))
  nn=n[draw].sum();a=sy[draw].sum();sst=sy2[draw].sum()-a*a/nn
  d.append(float((s0[draw].sum()-s1[draw].sum())/sst))
 d=np.array(d)
 return {"units":len(units),"replicates":reps,"seed":seed,
         "delta_r2_ci95":np.quantile(d,[.025,.5,.975]).tolist(),
         "positive_fraction":float((d>0).mean())}
def main():
 assert not (OUT/"ENSEMBLE_RESULTS.json").exists()
 protocol=json.loads((OUT/"REPLAY_PROTOCOL.json").read_text())
 seal=json.loads((OUT/"REPLAY_SEAL.json").read_text())
 assert seal["protocol_sha256"]==sha(OUT/"REPLAY_PROTOCOL.json")
 assert seal["source_hashes"][str(IDENTITY.relative_to(ROOT))]==sha(IDENTITY)
 review=json.loads((OUT/"REPLAY_REVIEW.json").read_text())
 replay=json.loads((OUT/"REPLAY_RESULT.json").read_text())
 assert review["passed"] and review["seal_sha256"]==sha(OUT/"REPLAY_SEAL.json")
 assert replay["seal_sha256"]==sha(OUT/"REPLAY_SEAL.json")
 paths={
  "mto_eta0":ETA/"mto_eta0/val_predictions.npz",
  "mto_eta01":ETA/"mto_eta01/val_predictions.npz",
  "mto_eta1":ETA/"mto_eta1/val_predictions.npz",
  "G1":OUT/"runtime/G1_selected_native_val.npz",
  "G3":OUT/"runtime/G3_selected_native_val.npz"}
 for key in ("mto_eta0","mto_eta01","mto_eta1"):
  assert sha(paths[key])==seal["source_hashes"][str(paths[key].relative_to(ROOT))]
 for key in ("G1","G3"):
  assert sha(paths[key])==replay["results"][key]["prediction_sha256"]
 arrays={key:load(path) for key,path in paths.items()}
 ref=arrays["mto_eta0"]
 for key,a in arrays.items():
  for field in ("ids","indices","f_true","mask_f_true"):
   assert np.array_equal(a[field],ref[field]),(key,field)
  assert a["f"].shape==ref["f_true"].shape
 y=np.asarray(ref["f_true"],dtype=np.float64);mask=np.asarray(ref["mask_f_true"],dtype=bool)
 assert y.shape==(6686,10) and mask.all()
 assert np.all(np.isfinite(y[mask]))
 ids=np.asarray(ref["ids"]);indices=np.asarray(ref["indices"])
 assert len(set(ids.tolist()))==6686
 pred={key:np.asarray(a["f"],dtype=np.float64) for key,a in arrays.items()}
 assert all(np.all(np.isfinite(v[mask])) for v in pred.values())
 pred["fixed_equal3"]=(pred["mto_eta0"]+pred["mto_eta01"]+pred["mto_eta1"])/3.0
 pred["G1_G3_equal2"]=(pred["G1"]+pred["G3"])/2.0
 pred["eta0_eta01_eta1_G1_G3_equal5"]=(pred["mto_eta0"]+pred["mto_eta01"]+pred["mto_eta1"]+pred["G1"]+pred["G3"])/5.0
 all_metrics={key:metrics(y,p,mask) for key,p in pred.items()}
 assert abs(all_metrics["mto_eta0"]["r2"]-0.4052941183410983)<1e-10
 assert abs(all_metrics["fixed_equal3"]["r2"]-0.4494214632744171)<1e-10
 for key in ("G1","G3"):
  assert abs(all_metrics[key]["r2"]-replay["results"][key]["native_fp64_trace_metric"]["r2"])<1e-12
 identity=json.loads(IDENTITY.read_text())
 grouped=defaultdict(list)
 for row,(id_,index) in enumerate(zip(ids,indices)):
  assert int(identity[int(index)][0])==int(id_)
  grouped[identity[int(index)][1]].append(row)
 groups=list(grouped.values())
 molecule_units=[[i] for i in range(len(ids))]
 q90,q99=np.quantile(y[mask],[.9,.99])
 comparisons={}
 for key in ("G1_G3_equal2","eta0_eta01_eta1_G1_G3_equal5"):
  p=pred[key];b=pred["fixed_equal3"]
  per_state=[]
  for state in range(10):
   s0=float(np.square(b[:,state]-y[:,state]).sum())
   s1=float(np.square(p[:,state]-y[:,state]).sum())
   per_state.append({"state":state+1,"state_index_zero_based":state,
                     "equal3_sse":s0,"candidate_sse":s1,
                     "delta_sse_candidate_minus_equal3":s1-s0,
                     "delta_r2_contribution":(s0-s1)/all_metrics["fixed_equal3"]["sst"]})
  tails={}
  for label,q in (("q90",q90),("q99",q99)):
   t=mask&(y>=q)
   tails[label]={"threshold":float(q),"count":int(t.sum()),
                 "equal3_sse":float(np.square(b[t]-y[t]).sum()),
                 "candidate_sse":float(np.square(p[t]-y[t]).sum())}
   tails[label]["delta_sse_candidate_minus_equal3"]=tails[label]["candidate_sse"]-tails[label]["equal3_sse"]
  comparisons[key]={"delta_r2_vs_equal3":all_metrics[key]["r2"]-all_metrics["fixed_equal3"]["r2"],
                    "delta_sse_vs_equal3":all_metrics[key]["sse"]-all_metrics["fixed_equal3"]["sse"],
                    "molecule_bootstrap":bootstrap(y,b,p,molecule_units),
                    "connectivity_bootstrap":bootstrap(y,b,p,groups),
                    "per_state":per_state,"tails":tails,
                    "model_count":2 if key=="G1_G3_equal2" else 5}
 report={"classification":"exploratory validation-only fixed native-f ensembles; post-individual-result choices",
         "protocol_sha256":sha(OUT/"REPLAY_PROTOCOL.json"),
         "ensemble_evaluator_sha256":sha(Path(__file__)),
         "replay_result_sha256":sha(OUT/"REPLAY_RESULT.json"),
         "source_prediction_sha256":{key:sha(path) for key,path in paths.items()},
         "identity_audit_sha256":sha(IDENTITY),
         "molecules":6686,"states":10,"connectivity_groups":len(groups),
         "metrics":all_metrics,"comparisons_vs_fixed_equal3":comparisons,
         "fixed_equal3_model_count":3,
         "limitations":["All base checkpoints and this combination were assessed on validation; bootstrap intervals are conditional on model selection and do not imply a fresh holdout.",
                        "Historical test was previously exposed elsewhere; no test arrays used here.",
                        "Inference requires two model forwards for G1+G3 or five for the equal-five, compared with three for equal-three; chan64 G1/G3 have about 3.45M/3.39M parameters, so forward counts are not calibrated latency ratios."]}
 p=OUT/"ENSEMBLE_RESULTS.json"
 p.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"path":str(p),"sha256":sha(p),"metrics":{k:all_metrics[k] for k in ("G1_G3_equal2","eta0_eta01_eta1_G1_G3_equal5","fixed_equal3")}},indent=2))
if __name__=="__main__":main()
