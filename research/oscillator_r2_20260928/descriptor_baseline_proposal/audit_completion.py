"""Saved-artifact-only audit of one completed descriptor probe; no model inference."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from array_io import member
from fit_worker import metric,ROOT,RUN,SOURCE,sha

def J(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def C(v,m):
 if not v:raise RuntimeError(m)
def close(a,b,tol=1e-10):C(abs(a-b)<=tol,"metric mismatch")
def main():
 out=ROOT/"completion_receipts";out.mkdir(exist_ok=True)
 C(not (RUN/"FAILED.json").exists() and not (RUN/"INCOMPLETE_RESOURCE.json").exists(),"failure receipt")
 q=J(RUN/"PROBE_COMPLETE.json");f=J(RUN/"FIT_FREEZE.json");v=J(RUN/"VALIDATION_RESULTS.json");review=J(ROOT/"IMPLEMENTATION_REVIEW.json");pre=J(ROOT/"IMPLEMENTATION_PREFLIGHT.json");auth=J(ROOT/"FIT_AUTHORIZATION.json");seal=J(ROOT/"SOURCE_SEAL.json");launch=J(RUN/"DESCRIPTOR_LAUNCH_RECEIPT.json");started=J(RUN/"SUPERVISOR_STARTED.json");train=J(RUN/"PHASE_TRAIN_RESOURCE.json");val=J(RUN/"PHASE_VALIDATION_RESOURCE.json");ts=J(RUN/"TRAIN_STARTED.json");vs=J(RUN/"VALIDATION_STARTED.json")
 C(q["event"]=="PROBE_COMPLETE" and f["event"]=="FIT_FREEZE" and v["event"]=="VALIDATION_COMPLETE" and not q["test_access"] and not f["test_access"] and not v["test_access"],"terminal events")
 C(review["passed"] and pre["passed"] and auth["approved"] and auth["one_run"],"review/auth")
 C(sha(ROOT/"IMPLEMENTATION_REVIEW.json")==q["review_sha256"]==launch["implementation_review_sha256"]==auth["review_sha256"],"review hash")
 C(sha(ROOT/"SOURCE_SEAL.json")==q["source_seal_sha256"]==f["source_seal_sha256"]==v["source_seal_sha256"]==auth["source_seal_sha256"],"seal hash")
 C(sha(ROOT/"IMPLEMENTATION_PREFLIGHT.json")==f["preflight_sha256"]==auth["preflight_sha256"],"preflight hash")
 for item in seal["files_pre_fit"]+seal["files_post_fit"]:C(sha(item["path"])==item["sha256"],"source drift "+item["path"])
 C(q["fit_freeze_sha256"]==sha(RUN/"FIT_FREEZE.json") and q["validation_results_sha256"]==sha(RUN/"VALIDATION_RESULTS.json") and vs["fit_freeze_sha256"]==sha(RUN/"FIT_FREEZE.json"),"phase hashes")
 C(ts["time"]<f["time"]<vs["time"]<v["time"],"train freeze before validation")
 C(q["phase_resource_receipts"]["train"]==sha(RUN/"PHASE_TRAIN_RESOURCE.json") and q["phase_resource_receipts"]["validation"]==sha(RUN/"PHASE_VALIDATION_RESOURCE.json"),"resource hashes")
 for phase,res in (("train",train),("validation",val)):
  C(res["returncode"]==0 and res["resource_failure"] is None and res["resource_limit_bytes"]==16*1024**3,"resource "+phase)
  C(res["elapsed_seconds"]<4*3600 and res["peak_tree_rss_bytes"]<16*1024**3 and len(res["admission"]["selected_cpus"])==8,"resource cap "+phase)
  C(res["admission"]["available_bytes"]>=20*1024**3,"memory admission "+phase)
 C(q["elapsed_seconds"]<4*3600 and not Path("/proc/"+str(started["pid"])).exists(),"supervisor terminal")
 C(launch["supervisor"]["pid"]==started["pid"] and launch["supervisor"]["start_ticks"]==started["start_ticks"] and launch["worker"]["pid"]==train["child_pid"],"launch identity")
 for name,file,key in (("train_features","TRAIN_FEATURES.npy","feature_hash"),("forest","FOREST.joblib","forest_hash"),("train_predictions","TRAIN_PREDICTIONS.npz","train_predictions_hash")):
  C(sha(RUN/file)==f[key],name+" hash")
 C(sha(RUN/"VAL_FEATURES.npy")==v["val_feature_hash"] and sha(RUN/"VAL_PREDICTIONS.npz")==v["val_predictions_hash"],"val artifact hashes")
 config=J(ROOT/"IMPLEMENTATION_CONFIG.json")
 C(all(f["effective_estimator_params"].get(k)==z for k,z in config["learner"].items()) and f["n_estimators"]==256,"estimator params")
 trainidx=np.asarray(member(SOURCE/"dataset.npz","train"));rawids=member(SOURCE/"raw_labels.npz","ids");rawf=member(SOURCE/"raw_labels.npz","f")
 C(hashlib.sha256(trainidx.tobytes()).hexdigest()==f["train_indices_sha256"] and hashlib.sha256(np.asarray(rawids[trainidx]).tobytes()).hexdigest()==f["train_ids_sha256"],"train identity")
 with np.load(RUN/"TRAIN_PREDICTIONS.npz",allow_pickle=False) as z:
  C(np.array_equal(z["indices"],trainidx) and z["truth"].dtype==np.float64 and np.array_equal(z["truth"],np.asarray(rawf[trainidx],dtype=np.float64)),"train raw truth")
  m=metric(z["truth"],z["pred"])
 for key in ("sse","sst","r2","mae","rmse"):close(m[key],f["train_metrics"][key],1e-9)
 import sys
 sys.path.insert(0,str(ROOT.parent/"fixed_seven_sidecar"));import fixed_seven as f7
 provenance,old,pins,rows=f7.static_pins();five,F5=f7.load_five(provenance,rows)
 with np.load(RUN/"VAL_PREDICTIONS.npz",allow_pickle=False) as z:
  C(np.array_equal(z["ids"],rows["ids"]) and np.array_equal(z["indices"],rows["indices"]),"validation IDs/indices")
  C(z["f_true"].dtype==np.float64 and np.array_equal(z["f_true"],rows["truth"]) and np.array_equal(z["mask_f_true"],rows["mask"]),"validation FP64 truth/mask")
  D=z["D"].copy();blend=z["blend"].copy()
 C(D.shape==(6686,10) and np.isfinite(D).all() and (D>=0).all() and np.allclose(blend,(D+F5)/2,rtol=0,atol=1e-15),"fixed half blend")
 metrics={}
 for name,pred in (("D",D),("F5",F5),("equal_blend",blend)):
  m=metric(rows["truth"],pred);metrics[name]=m
  for key in ("sse","sst","r2","mae","rmse"):close(m[key],v["metrics"][name][key],1e-9)
  signed=(pred-rows["truth"]);sm=v["signed_error_pred_minus_truth"][name]
  close(float(signed.mean()),sm["overall_mean"],1e-12)
  C(np.allclose(signed.mean(axis=0),sm["per_state_mean"],atol=1e-12,rtol=0),"signed states")
 output={"passed":True,"classification":"one completed fixed descriptor fit/validation; saved arrays and receipts only; no model inference or test","source_files_verified":len(seal["files_pre_fit"])+len(seal["files_post_fit"]),"terminal_sha256":sha(RUN/"PROBE_COMPLETE.json"),"review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),"authorization_sha256":sha(ROOT/"FIT_AUTHORIZATION.json"),"fit_freeze_sha256":sha(RUN/"FIT_FREEZE.json"),"validation_results_sha256":sha(RUN/"VALIDATION_RESULTS.json"),"metrics":metrics,"train_metrics":f["train_metrics"],"resource":{"total_seconds":q["elapsed_seconds"],"train_peak_rss_bytes":train["peak_tree_rss_bytes"],"val_peak_rss_bytes":val["peak_tree_rss_bytes"],"train_feature_seconds":f["train_feature_seconds"],"forest_fit_seconds":f["fit_seconds"],"val_feature_seconds":v["validation_feature_seconds"]},"limits":["Validation and train predictions are historical saved arrays; this audit ran no forest prediction or model inference.","Fixed descriptor recipe and equal half-blend both underperform F5 on reused validation; no promotion/test/retuning."]}
 p=out/"DESCRIPTOR_COMPLETION_AUDIT.json";p.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 md=["# Descriptor baseline completion audit","","PASS: one reviewed 803-feature, 256-tree fit completed; fit froze before validation, source/input hashes and raw FP64 truth match. No test or audit-time inference.","","| Predictor | Raw native-f R² | SSE | MAE |","|---|---:|---:|---:|"]
 for k in ("D","F5","equal_blend"):
  z=metrics[k];md.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(k,z["r2"],z["sse"],z["mae"]))
 md += ["","Train D R² {:.6f}; validation D R² {:.6f}. Equal half-blend R² {:.6f} is below frozen F5 {:.6f}.".format(f["train_metrics"]["r2"],metrics["D"]["r2"],metrics["equal_blend"]["r2"],metrics["F5"]["r2"]),"No model promotion, additional fit, validation search or test comparison follows from this result."]
 (out/"DESCRIPTOR_COMPLETION_AUDIT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
 print(json.dumps({"passed":True,"D_r2":metrics["D"]["r2"],"blend_r2":metrics["equal_blend"]["r2"],"audit_sha256":sha(p)}))
if __name__=="__main__":main()
