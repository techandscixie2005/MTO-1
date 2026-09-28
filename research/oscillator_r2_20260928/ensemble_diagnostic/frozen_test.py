#!/usr/bin/env python3
"""Freeze validation-selected equal eta ensemble, then evaluate historical test once."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path("/home/inspur/MTO-1")
OUT=ROOT/"research/oscillator_r2_20260928/ensemble_diagnostic"
SRC=ROOT/"experiments/qm9s_eta_Ef_20260926"
NAMES=("mto_eta0","mto_eta01","mto_eta1")
KEYS=("ids","indices","f","f_true","mask_f_true")
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def write_once(path,obj):
    assert not path.exists(),"Refuse to overwrite frozen/evaluated artifact: "+str(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,sort_keys=True)+"\n")
    os.replace(tmp,path)
def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.array(z[k],copy=True) for k in KEYS}
def score(y,p,m):
    yy=y[m];pp=p[m];e=pp-yy
    sse=float(np.square(e).sum());sst=float(np.square(yy-yy.mean()).sum())
    return {"count":int(len(yy)),"sse":sse,"sst":sst,"r2":float(1-sse/sst),
            "rmse":float(np.sqrt(np.mean(np.square(e)))),
            "mae":float(np.mean(np.abs(e)))}
def aligned(arrays):
    a=arrays[NAMES[0]]
    for name in NAMES[1:]:
        b=arrays[name]
        for key in ("ids","indices","f_true","mask_f_true"):
            assert np.array_equal(a[key],b[key]),"Misaligned "+key+" "+name
    ids=a["ids"]
    assert ids.shape==(6686,) and len(np.unique(ids))==6686
    for name in NAMES:
        assert arrays[name]["f"].shape==(6686,10)
        assert arrays[name]["f_true"].shape==(6686,10)
        assert arrays[name]["mask_f_true"].shape==(6686,10)
    return a
def freeze():
    diagnostic=json.loads((OUT/"ensemble_metrics.json").read_text())
    assert diagnostic["split"]=="validation only"
    valpaths={n:SRC/"runs"/n/"val_predictions.npz" for n in NAMES}
    for n,p in valpaths.items():
        assert sha(p)==diagnostic["source_files"][n]["sha256"]
    arrays={n:load(p) for n,p in valpaths.items()}
    a=aligned(arrays);truth=a["f_true"].astype(np.float64);m=a["mask_f_true"].astype(bool)
    pred=np.mean(np.stack([arrays[n]["f"].astype(np.float64) for n in NAMES]),axis=0)
    base=arrays["mto_eta0"]["f"].astype(np.float64)
    bm=score(truth,base,m);em=score(truth,pred,m)
    assert abs(bm["r2"]-0.4052941183410983)<1e-12
    assert abs(em["r2"]-0.44942146327441623)<1e-12
    select=json.loads((SRC/"reports/selection_before_test.json").read_text())
    checkpoints={n:{"path":str(SRC/"runs"/n/"best.pt"),
                    "sha256":select["checkpoints"][n]["best_pt_sha256"],
                    "best_epoch":select["checkpoints"][n]["best_epoch"],
                    "selection":"own validation L_eta"} for n in NAMES}
    report={"time":time.time(),"status":"FROZEN_BEFORE_THIS_TEST_STAGE",
      "hypothesis":"Average three existing native oscillator-strength predictions to reduce complementary errors.",
      "classification":"Exploratory: historical test split was evaluated in prior campaigns; this script has not opened test data during freeze.",
      "selected":"all_three_equal","weights":{n:1/3 for n in NAMES},
      "prediction_rule":"Arithmetic mean of saved native f arrays. Do not average E/A and derive f.",
      "selection_evidence":"Best fixed candidate on validation pooled raw-f R2 among recorded single and equal-weight ensemble candidates; no fitted weights.",
      "source_validation_paths":{n:str(p) for n,p in valpaths.items()},
      "source_validation_sha256":{n:sha(p) for n,p in valpaths.items()},
      "source_checkpoint_provenance":checkpoints,
      "source_selection_report_sha256":sha(SRC/"reports/selection_before_test.json"),
      "validation_diagnostic_sha256":sha(OUT/"ensemble_metrics.json"),
      "validation_protocol_sha256":sha(OUT/"PROTOCOL.json"),
      "ensemble_diagnostic_code_sha256":sha(OUT/"diagnose.py"),
      "evaluation_code_sha256":sha(Path(__file__)),
      "validation_baseline":bm,"validation_equal_three":em,
      "validation_delta_r2":em["r2"]-bm["r2"],
      "validation_relative_sse_reduction":(bm["sse"]-em["sse"])/bm["sse"],
      "bright_thresholds_raw_f_from_validation":{"q90":float(np.quantile(truth[m],.9)),
                                                 "q99":float(np.quantile(truth[m],.99))},
      "test_paths_fixed":{n:str(SRC/"runs"/n/"test_predictions.npz") for n in NAMES},
      "test_metrics_opened_during_freeze":False,
      "test_evaluation_rule":"One run after freeze; exact alignment, validation/test disjointness; pooled raw f R2/MAE/RMSE; paired molecule bootstrap 2000 seed 20260928; per-state and validation-defined q90/q99 bright SSE.",
      "ensemble_inference_cost":"Three separately trained model predictions rather than one model."}
    write_once(OUT/"FROZEN_CHOICE.json",report)
    print(json.dumps({"frozen_choice_sha256":sha(OUT/"FROZEN_CHOICE.json"),
                      "validation_baseline_r2":bm["r2"],
                      "validation_equal_three_r2":em["r2"],
                      "q90":report["bright_thresholds_raw_f_from_validation"]["q90"],
                      "q99":report["bright_thresholds_raw_f_from_validation"]["q99"]}))
def evaluate():
    frozen_path=OUT/"FROZEN_CHOICE.json"
    f=json.loads(frozen_path.read_text())
    assert f["selected"]=="all_three_equal" and f["evaluation_code_sha256"]==sha(Path(__file__))
    assert not (OUT/"EXPLORATORY_TEST.json").exists()
    arrays={n:load(Path(f["test_paths_fixed"][n])) for n in NAMES}
    a=aligned(arrays)
    with np.load(f["source_validation_paths"]["mto_eta0"],allow_pickle=False) as z:
        validx=z["indices"];valid=z["ids"]
    assert np.intersect1d(validx,a["indices"]).size==0
    assert np.intersect1d(valid,a["ids"]).size==0
    truth=a["f_true"].astype(np.float64);mask=a["mask_f_true"].astype(bool)
    baseline=arrays["mto_eta0"]["f"].astype(np.float64)
    ensemble=np.mean(np.stack([arrays[n]["f"].astype(np.float64) for n in NAMES]),axis=0)
    assert mask.all() and np.isfinite(truth).all() and np.isfinite(ensemble).all()
    bm=score(truth,baseline,mask);em=score(truth,ensemble,mask)
    per_state=[]
    for j in range(10):
        m=np.zeros_like(mask);m[:,j]=mask[:,j]
        per_state.append({"state":j+1,"baseline":score(truth,baseline,m),
                          "ensemble":score(truth,ensemble,m)})
    bright={}
    for label,threshold in f["bright_thresholds_raw_f_from_validation"].items():
        m=mask&(truth>=threshold)
        bright[label]={"threshold":threshold,"baseline":score(truth,baseline,m),
                       "ensemble":score(truth,ensemble,m),
                       "sse_reduction":float(np.square(baseline[m]-truth[m]).sum()-
                                             np.square(ensemble[m]-truth[m]).sum())}
    rng=np.random.default_rng(20260928)
    deltas=[]
    for _ in range(2000):
        idx=rng.integers(0,6686,6686)
        y=truth[idx];p0=baseline[idx];p1=ensemble[idx];mm=mask[idx]
        yy=y[mm];sst=np.square(yy-yy.mean()).sum()
        deltas.append(float((np.square(p0[mm]-yy).sum()-np.square(p1[mm]-yy).sum())/sst))
    deltas=np.asarray(deltas)
    report={"time":time.time(),
      "classification":"Single exploratory historical-test comparison after frozen validation choice; historical test previously exposed.",
      "frozen_choice_sha256":sha(frozen_path),
      "test_source_sha256":{n:sha(Path(f["test_paths_fixed"][n])) for n in NAMES},
      "alignment":{"exact_ids_indices_raw_truth_masks_match":True,
                   "molecules":6686,"states":10,"validation_ids_and_indices_disjoint":True},
      "prediction_rule":f["prediction_rule"],
      "baseline":bm,"ensemble":em,
      "delta_r2":em["r2"]-bm["r2"],
      "relative_sse_reduction":(bm["sse"]-em["sse"])/bm["sse"],
      "bootstrap":{"resampling_unit":"molecule with ten states","replicates":2000,
        "seed":20260928,"paired_delta_r2_2p5_50_97p5":np.quantile(deltas,[.025,.5,.975]).tolist(),
        "positive_fraction":float((deltas>0).mean())},
      "per_state":per_state,"bright":bright,
      "test_driven_reweighting":False,
      "inference_cost":"Three model forward passes; saved predictions used here."}
    write_once(OUT/"EXPLORATORY_TEST.json",report)
    print(json.dumps({"baseline":bm,"ensemble":em,"delta_r2":report["delta_r2"],
                      "bootstrap":report["bootstrap"],
                      "test_report_sha256":sha(OUT/"EXPLORATORY_TEST.json")}))
if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in ("freeze","evaluate")
    globals()[sys.argv[1]]()

