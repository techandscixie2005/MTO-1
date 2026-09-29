"""One frozen seven-way scratch comparison from completed saved validation arrays."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928")
HERE=ROOT/"fixed_scratch_seven"
sys.path.insert(0,str(ROOT/"fixed_seven_sidecar"))
import fixed_seven as f7
sys.path.insert(0,str(ROOT/"fixed_seven_completion"))
import seed_mask_compat as compat
AMEND_SHA="5c315badf8c93408693250453197dec6cecaeb0b6f89c193adb1f8f4af30b364"
PROV_SHA="dbb69f131508b21ff22f67ebe1c1ee0ba97ac855ccb2731bdfab3e679491f16c"

def need(v,m):
 if not v:raise RuntimeError(m)
def J(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def static():
 need(f7.sha(ROOT/"postrun_geometry/FIXED_SCRATCH_SEVEN_SECONDARY_AMENDMENT.md")==AMEND_SHA,"amendment hash")
 need(f7.sha(ROOT/"postrun_geometry/FIXED_SCRATCH_SEVEN_SECONDARY_PROVENANCE.json")==PROV_SHA,"provenance hash")
 p=J(ROOT/"postrun_geometry/FIXED_SCRATCH_SEVEN_SECONDARY_PROVENANCE.json")
 need(p["amendment_sha256"]==AMEND_SHA,"amendment link")
 for rel,digest in p["reference_hashes"].items():need(f7.sha(ROOT/rel)==digest,"reference drift "+rel)
 for item in p["frozen_five_components"].values():need(f7.sha(item["prediction_path"])==item["prediction_sha256"],"F5 component drift")
 _,old,pins,rows=f7.static_pins();five,F5=f7.load_five(p,rows)
 return p,rows,five,F5
def reviewed():
 p=HERE/"IMPLEMENTATION_REVIEW.json";need(p.is_file(),"PENDING: independent scratch-seven review")
 d=J(p);need(d.get("passed") is True and d.get("script_sha256")==f7.sha(__file__) and d.get("protocol_sha256")==f7.sha(HERE/"PROTOCOL.md") and d.get("preflight_sha256")==f7.sha(HERE/"PREFLIGHT.json") and d.get("amendment_sha256")==AMEND_SHA and d.get("provenance_sha256")==PROV_SHA and d.get("fixed7_source_sha256")==f7.sha(ROOT/"fixed_seven_sidecar/fixed_seven.py") and d.get("compat_source_sha256")==f7.sha(ROOT/"fixed_seven_completion/seed_mask_compat.py"),"exact scratch-seven review")
 f7.reviewed_execution_gate();compat.reviewed_gate();return d
def selected_scratch_compat(base,fit,rows,arm):
 fixed_path=base/"FIXED_CHECKPOINT_METRICS.json"
 need(fixed_path.is_file() and f7.sha(fixed_path)==fit["fixed_metrics_sha256"],"scratch "+arm+" fixed metrics hash")
 fixed=J(fixed_path);epoch=fit["best"]["raw_f"]["epoch"]
 need(fixed["raw_f"]["epoch"]==epoch,"scratch "+arm+" selected epoch")
 ckpt=base/"best_raw_f.pt"
 need(ckpt.is_file() and f7.sha(ckpt)==fit["checkpoint_hashes"]["raw_f"] and fixed["raw_f"]["checkpoint_sha256"]==fit["checkpoint_hashes"]["raw_f"],"scratch "+arm+" selected checkpoint")
 path=base/"val_best_raw_f.npz";expected=fit["prediction_hashes"]["best_raw_f"]
 need(path.is_file() and f7.sha(path)==expected,"scratch "+arm+" selected prediction hash")
 # The scratch writer omits mask_f_true. Bind it to the pinned eta0 array
 # and raw-label-backed validation mask; do not infer validity from predictions.
 ref=ROOT.parent.parent/"experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz"
 need(f7.sha(ref)==fit["source_hashes"][str(ref)],"scratch "+arm+" authoritative mask source hash")
 with np.load(ref,allow_pickle=False) as z:
  need("mask_f_true" in z.files and z["mask_f_true"].dtype==np.bool_ and z["mask_f_true"].all() and rows["mask"].dtype==np.bool_ and rows["mask"].all() and np.array_equal(z["mask_f_true"],rows["mask"]),"authoritative mask mismatch")
  need(np.array_equal(z["indices"],rows["indices"]) and np.array_equal(z["ids"],rows["ids"]) and z["f_true"].dtype==np.float64 and np.array_equal(z["f_true"],rows["truth"]),"authoritative raw truth alignment")
  energy=z["E_true"].copy()
 with np.load(path,allow_pickle=False) as z:
  need("E_true" in z.files and z["E_true"].dtype==np.float64 and np.array_equal(z["E_true"],energy),"scratch "+arm+" raw E truth")
  if "mask_f_true" in z.files:need(z["mask_f_true"].dtype==np.bool_ and np.array_equal(z["mask_f_true"],rows["mask"]),"scratch "+arm+" optional mask")
 pred=f7.pg.load_pred(path,rows,expected)
 observed=f7.metric(rows["truth"],pred)["sse"]
 need(abs(observed-fit["best"]["raw_f"]["metric"])<=1e-8,"scratch "+arm+" selected raw SSE")
 return pred,{"fixed_metrics_sha256":fit["fixed_metrics_sha256"],"raw_f_epoch":epoch,"selected_checkpoint_sha256":fit["checkpoint_hashes"]["raw_f"],"selected_prediction_sha256":expected,"saved_array_sse":observed,"mask_source_sha256":f7.sha(ref)}
def scratch_artifacts_after_terminal_compat(project,rows):
 fits=f7.scratch_terminal_pair(project)
 out={};evidence={}
 for arm in ("native","mto"):
  pred,details=selected_scratch_compat(project/"runs"/arm,fits[arm]["receipt"],rows,arm)
  out[arm]=pred;evidence[arm]={"terminal_sha256":fits[arm]["terminal_sha256"],**details}
 return out,evidence,fits
def scratch_completed(rows,p):
 s=ROOT/"scratch_readout_comparison"
 # Complete both terminal gates before reading any scratch prediction.
 fits=f7.scratch_terminal_pair(s)
 native=p["native_completed_component"]
 need(fits["native"]["terminal_sha256"]==native["receipt_sha256"],"frozen native terminal changed")
 pins=J(ROOT/"fixed_seven_sidecar/SCRATCH_SOURCE_PINS.json")
 need(f7.sha(ROOT/"fixed_seven_sidecar/SCRATCH_SOURCE_PINS.json")==f7.SCRATCH_PINS_SHA,"scratch source pins")
 for rel,digest in pins["scratch_source_hashes"].items():need(f7.sha(s/rel)==digest,"scratch source pin "+rel)
 review=J(s/"IMPLEMENTATION_REVIEW.json");pre=J(s/"PREFLIGHT.json")
 need(review["passed"] and pre["passed"] and review["preflight_sha256"]==f7.sha(s/"PREFLIGHT.json") and pre["source_hashes"]==review["source_hashes"] and pre["code_hashes"]==review["code_hashes"],"scratch reviewed maps")
 f7.verify_file_map(review["source_hashes"]);f7.verify_file_map(review["code_hashes"])
 for arm in ("native","mto"):
  fit=fits[arm]["receipt"]
  need(fit["source_hashes"]==review["source_hashes"] and fit["code_hashes"]==review["code_hashes"],"scratch receipt source "+arm)
 need(fits["native"]["receipt"]["best"]["raw_f"]["epoch"]==native["raw_f_selected_epoch"] and fits["native"]["receipt"]["checkpoint_hashes"]["raw_f"]==native["checkpoint_sha256"] and fits["native"]["receipt"]["prediction_hashes"]["best_raw_f"]==native["prediction_sha256"],"native selected pin")
 out={};evidence={}
 for arm in ("native","mto"):
  fit=fits[arm]["receipt"]
  pred,details=selected_scratch_compat(s/"runs"/arm,fit,rows,arm)
  out[arm]=pred;evidence[arm]={"terminal_sha256":fits[arm]["terminal_sha256"],**details}
 return out,evidence
def seed_reference(rows,five,F5):
 s=ROOT/"eta0_seed_replication"
 terminals=[s/f"runs/seed{k}/FIT_COMPLETE.json" for k in (23,37)]
 if not all(p.is_file() for p in terminals):return None,{"status":"PENDING: both genuine seed terminals required"}
 rev=J(s/"IMPLEMENTATION_REVIEW.json");pre=J(s/"PREFLIGHT.json")
 need(rev["passed"] and pre["passed"] and rev["preflight_sha256"]==f7.sha(s/"PREFLIGHT.json") and pre["source_hashes"]==rev["source_hashes"] and pre["code_hashes"]==rev["code_hashes"],"seed reviewed maps")
 f7.verify_file_map(rev["source_hashes"]);f7.verify_file_map(rev["code_hashes"])
 fits=f7.seed_terminal_pair(s)
 for k in (23,37):need(fits[k]["receipt"]["source_hashes"]==rev["source_hashes"] and fits[k]["receipt"]["code_hashes"]==rev["code_hashes"],"seed receipt source")
 ref=ROOT.parent.parent/"experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz"
 with np.load(ref,allow_pickle=False) as z:
  need(z["mask_f_true"].dtype==np.bool_ and z["mask_f_true"].all() and np.array_equal(z["indices"],rows["indices"]) and np.array_equal(z["ids"],rows["ids"]),"seed reference mask/alignment")
  et=z["E_true"].copy();need(et.dtype==np.float64,"seed E truth dtype")
 seeds={};evidence={}
 for k in (23,37):
  fit=fits[k]["receipt"];base=s/f"runs/seed{k}"
  fixed=J(base/"FIXED_CHECKPOINT_METRICS.json")
  need(f7.sha(base/"FIXED_CHECKPOINT_METRICS.json")==fit["fixed_metrics_sha256"] and fixed["legacy"]["epoch"]==fit["best_legacy_epoch"],"seed fixed selected")
  need(f7.sha(base/"best_legacy.pt")==fit["checkpoint_hashes"]["legacy"],"seed checkpoint")
  path=base/"val_best_legacy.npz"
  seeds[k]=compat.load_seed(path,{**rows,"E_true":et},fit["val_prediction_hashes"]["best_legacy"])
  need(abs(f7.metric(rows["truth"],seeds[k])["sse"]-fixed["legacy"]["val"]["raw_native_f"]["sse"])<=1e-3,"seed fixed SSE")
  evidence[str(k)]={"terminal_sha256":fits[k]["terminal_sha256"],"legacy_epoch":fit["best_legacy_epoch"],"checkpoint_sha256":fit["checkpoint_hashes"]["legacy"],"prediction_sha256":fit["val_prediction_hashes"]["best_legacy"]}
 seed7=(5*F5+seeds[23]+seeds[37])/7
 prior_path=ROOT/"fixed_seven_sidecar/SEED_FIXED7_RESULTS.json"
 execution=J(ROOT/"fixed_seven_completion/SEED_COMPAT_EXECUTION.json")
 need(execution["result_sha256"]==f7.sha(prior_path),"prior F7_seed execution/result binding")
 need(execution["adapter_sha256"]==f7.sha(ROOT/"fixed_seven_completion/seed_mask_compat.py") and execution["test_access"] is False and execution["inference"] is False,"prior F7_seed adapter binding")
 prior=J(prior_path);observed=f7.metric(rows["truth"],seed7)
 need(abs(observed["r2"]-prior["metrics"]["F7"]["r2"])<1e-12 and abs(observed["sse"]-prior["metrics"]["F7"]["sse"])<1e-9,"prior F7_seed reproduction")
 return seed7,evidence
def signed(y,p):
 d=p-y;return {"overall_mean":float(d.mean()),"per_state_mean":[float(x) for x in d.mean(axis=0)]}
def completed():
 reviewed();p,rows,five,F5=static()
 output=HERE/"F7_SCRATCH_RESULTS.json";report=HERE/"F7_SCRATCH_REPORT.md"
 need(not output.exists() and not report.exists(),"completed output exists")
 scratch,evidence=scratch_completed(rows,p)
 native,mto=scratch["native"],scratch["mto"]
 F7=(5*F5+native+mto)/7
 need(np.allclose(F7,(sum(five.values())+native+mto)/7,rtol=0,atol=1e-15),"seven individual weights")
 seed7,seed_evidence=seed_reference(rows,five,F5)
 group_rows,group_keys=f7.identity_groups(rows)
 comparisons={"F7_scratch_minus_F5":f7.compare(rows,F5,F7,group_rows,group_keys)}
 metrics={"F5":f7.metric(rows["truth"],F5),"F7_scratch":f7.metric(rows["truth"],F7)}
 preds={"F5":F5,"F7_scratch":F7}
 if seed7 is not None:
  comparisons["F7_scratch_minus_F7_seed_legacy"]=f7.compare(rows,seed7,F7,group_rows,group_keys)
  metrics["F7_seed_legacy"]=f7.metric(rows["truth"],seed7);preds["F7_seed_legacy"]=seed7
 result={"classification":"exploratory pre-outcome-fixed scratch seven saved-validation secondary; no inference/test","time":time.time(),"protocol_sha256":f7.sha(HERE/"PROTOCOL.md"),"script_sha256":f7.sha(__file__),"review_sha256":f7.sha(HERE/"IMPLEMENTATION_REVIEW.json"),"amendment_sha256":AMEND_SHA,"provenance_sha256":PROV_SHA,"scratch_selected_evidence":evidence,"seed_reference_evidence":seed_evidence,"weights":"exact equal 1/7 over eta0,eta01,eta1,G1,G3,scratch_native_raw_f,scratch_mto_raw_f","inference_forward_counts":{"F5":5,"F7_scratch":7,"F7_seed_legacy":7 if seed7 is not None else None},"metrics":metrics,"signed_error_pred_minus_truth":{k:signed(rows["truth"],z) for k,z in preds.items()},"comparisons":comparisons,"connectivity_groups":len(group_rows),"fixed_q2_boundaries_train_only":rows["bounds2"].tolist(),"fixed_tail_thresholds":{"q90":f7.Q90,"q99":f7.Q99},"available_model_cost":{"scratch_native_checkpoint_bytes":(ROOT/"scratch_readout_comparison/runs/native/best_raw_f.pt").stat().st_size,"scratch_mto_checkpoint_bytes":(ROOT/"scratch_readout_comparison/runs/mto/best_raw_f.pt").stat().st_size},"limitations":["Post-native-result, pre-MTO-outcome exploratory fixed mixture on reused validation.","Bootstrap intervals omit adaptive model selection uncertainty.","Seven versus five/seven model forwards are counts, not calibrated latency or equal training cost.","Scratch models differ from legacy seeds in objective, selection, schedule and initialization; complementarity is not pure architecture causality.","No model inference, fitting, test access, filtering or alternative mixture."]}
 f7.write_json(output,result)
 lines=["# Fixed scratch-seven saved-array secondary","","Frozen equal seven individual native-f predictions; all 6,686 molecules and 66,860 states retained. No inference, fit or test.","","| Predictor | SSE | R² | RMSE | MAE |","|---|---:|---:|---:|---:|"]
 for k,m in metrics.items():lines.append("| {} | {:.9f} | {:.9f} | {:.7f} | {:.7f} |".format(k,m["sse"],m["r2"],m["rmse"],m["mae"]))
 for k,c in comparisons.items():lines.extend(["","## "+k,"","ΔR² {:+.9f}; ΔSSE {:+.9f}.".format(c["delta_r2_candidate_minus_reference"],c["delta_sse_candidate_minus_reference"]),"Molecule CI {}; connectivity-group CI {}.".format(c["molecule_bootstrap"]["delta_r2_ci95"],c["connectivity_bootstrap"]["delta_r2_ci95"])])
 lines.extend(["","Per-state, fixed tail, signed error and absolute concentration records are in JSON.","Native-minus-MTO remains the original scratch primary contrast. F5 and F7_seed are fixed supplemental references.","Validation has been reused; intervals are descriptive and no test is opened. Seven model forwards are not calibrated latency."])
 report.write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"F7_scratch_r2":metrics["F7_scratch"]["r2"],"F5_r2":metrics["F5"]["r2"],"seed_reference":seed7 is not None}))
def fixture():
 p,rows,five,F5=static()
 y=rows["truth"]
 need(abs(f7.metric(y,F5)["r2"]-f7.EXPECTED_F5_R2)<1e-11,"F5 reproduction")
 # Synthetic arrays only. This algebra is independent of real scratch/seed outcomes.
 a=F5+.01;b=F5-.02;s=F5+.03;t=F5-.04
 new=(5*F5+a+b)/7;seed=(5*F5+s+t)/7
 need(np.allclose(new,(sum(five.values())+a+b)/7,rtol=0,atol=1e-15),"seven arithmetic")
 gr,gk=f7.identity_groups(rows);c=f7.compare(rows,F5,new,gr,gk)
 need(c["molecule_bootstrap"]["replicates"]==c["connectivity_bootstrap"]["replicates"]==2000,"bootstrap units")
 # Only synthetic temporary terminal receipts are supplied; no active scratch prediction is opened.
 import tempfile
 with tempfile.TemporaryDirectory() as td:
  fake=Path(td)
  try:f7.scratch_terminal_pair(fake)
  except RuntimeError as e:need("FIT_COMPLETE" in str(e),"missing pair classification")
  else:raise RuntimeError("missing scratch terminals accepted")
  for arm in ("native","mto"):
   q=fake/"runs"/arm;q.mkdir(parents=True,exist_ok=True)
   (q/"FIT_COMPLETE.json").write_text(json.dumps({"event":"FIT_COMPLETE","arm":arm,"epochs":100,"steps":188100,"test_batches":0,"best":{"raw_f":{"epoch":21},"joint":{"epoch":21}}}))
  need(set(f7.scratch_terminal_pair(fake))=={"native","mto"},"synthetic pair")
  original_loader=globals()["selected_scratch_compat"];opened=[]
  def guarded_loader(*args,**kwargs):
   opened.append(args[3]);return np.zeros_like(y),{}
  globals()["selected_scratch_compat"]=guarded_loader
  try:
   (fake/"runs/mto/FIT_COMPLETE.json").unlink()
   try:scratch_artifacts_after_terminal_compat(fake,rows)
   except RuntimeError as e:need("FIT_COMPLETE" in str(e) and not opened,"missing second terminal opened prediction")
   else:raise RuntimeError("missing second terminal accepted")
   (fake/"runs/mto/FIT_COMPLETE.json").write_text(json.dumps({"event":"FIT_COMPLETE","arm":"mto","epochs":100,"steps":188100,"test_batches":0,"best":{"raw_f":{"epoch":21},"joint":{"epoch":21}}}))
   (fake/"runs/mto/INVALID.json").write_text("{}")
   try:scratch_artifacts_after_terminal_compat(fake,rows)
   except RuntimeError as e:need("FAILED/INVALID" in str(e) and not opened,"invalid second terminal opened prediction")
   else:raise RuntimeError("invalid scratch accepted")
   (fake/"runs/mto/INVALID.json").unlink()
   out,_,_=scratch_artifacts_after_terminal_compat(fake,rows)
   need(opened==["native","mto"] and set(out)=={"native","mto"},"valid synthetic pair loader sequence")
  finally:globals()["selected_scratch_compat"]=original_loader
 # Exercise the actual isolated scratch schema loader with synthetic files.
 with tempfile.TemporaryDirectory() as td:
  fake=Path(td);ref=ROOT.parent.parent/"experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz"
  with np.load(ref,allow_pickle=False) as z:energy=z["E_true"].copy()
  ck=fake/"best_raw_f.pt";ck.write_bytes(b"synthetic-selected-checkpoint")
  ckhash=f7.sha(ck);fixed=fake/"FIXED_CHECKPOINT_METRICS.json"
  fixed.write_text(json.dumps({"raw_f":{"epoch":21,"checkpoint_sha256":ckhash}}))
  path=fake/"val_best_raw_f.npz"
  base={"indices":rows["indices"],"ids":rows["ids"],"f_true":y,"E_true":energy,"f":F5}
  def write(**changes):
   fields={**base,**changes};np.savez(path,**fields)
  write()
  fit={"fixed_metrics_sha256":f7.sha(fixed),"best":{"raw_f":{"epoch":21,"metric":f7.metric(y,F5)["sse"]}},"checkpoint_hashes":{"raw_f":ckhash},"prediction_hashes":{"best_raw_f":f7.sha(path)},"source_hashes":{str(ref):f7.sha(ref)}}
  loaded,_=selected_scratch_compat(fake,fit,rows,"synthetic")
  need(np.array_equal(loaded,F5),"omitted-mask scratch schema failed")
  def rejects(label,fields=None,fit_change=None,row_change=None):
   if fields is not None:write(**fields)
   case={**fit}
   if fields is not None and label!="wrong array hash":case["prediction_hashes"]={"best_raw_f":f7.sha(path)}
   if fit_change:case={**case,**fit_change}
   rr={**rows,**(row_change or {})}
   try:selected_scratch_compat(fake,case,rr,"synthetic")
   except RuntimeError:return
   raise RuntimeError("synthetic invalid "+label+" accepted")
  rejects("wrong array hash",{"ids":rows["ids"]+1})
  write();fit["prediction_hashes"]["best_raw_f"]=f7.sha(path)
  rejects("wrong IDs",{"ids":rows["ids"]+1})
  write(f_true=y+1);fit["prediction_hashes"]["best_raw_f"]=f7.sha(path)
  rejects("wrong raw f",None)
  write(E_true=energy+1);fit["prediction_hashes"]["best_raw_f"]=f7.sha(path)
  rejects("wrong raw E",None)
  write(mask_f_true=np.zeros_like(rows["mask"]));fit["prediction_hashes"]["best_raw_f"]=f7.sha(path)
  rejects("false optional mask",None)
  write();fit["prediction_hashes"]["best_raw_f"]=f7.sha(path)
  rejects("false authoritative mask",None,row_change={"mask":np.zeros_like(rows["mask"])})
  rejects("wrong checkpoint hash",None,fit_change={"checkpoint_hashes":{"raw_f":"0"*64}})
 out={"passed":True,"classification":"frozen F5 arrays plus synthetic scratch/seed values only; no actual scratch/seed outcome reads","time":time.time(),"script_sha256":f7.sha(__file__),"protocol_sha256":f7.sha(HERE/"PROTOCOL.md"),"amendment_sha256":AMEND_SHA,"provenance_sha256":PROV_SHA,"F5_r2":f7.metric(y,F5)["r2"],"synthetic_seven_r2":f7.metric(y,new)["r2"],"synthetic_pair_r2":f7.metric(y,seed)["r2"],"synthetic_molecule_and_group_bootstrap":True,"missing_invalid_terminal_rejection":True,"selected_loader_called_only_after_both_terminals":True,"actual_compat_loader_synthetic_schema_and_negative_cases":True,"actual_scratch_or_seed_outcome_access":False,"test_access":False}
 (HERE/"PREFLIGHT.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 print(json.dumps({"passed":True,"F5_r2":out["F5_r2"]}))
if __name__=="__main__":
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument("--fixture",action="store_true");g.add_argument("--scratch-completed",action="store_true");args=a.parse_args()
 if args.fixture:fixture()
 else:completed()
