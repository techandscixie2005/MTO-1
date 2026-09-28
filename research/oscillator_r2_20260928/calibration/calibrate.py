#!/usr/bin/env python3
"""Validation-only cross-fit selection, then one exploratory test evaluation."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0")
CASES = ["identity", "scale", "per_state_025", "per_state_050",
         "per_state_075", "per_state_100", "affine_clip"]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def load(split):
    p = SRC / (split + "_predictions.npz")
    z = np.load(p)
    ids = z["ids"]
    x = z["f"].astype(np.float64)
    y = z["f_true"].astype(np.float64)
    mask = z["mask_f_true"]
    assert x.shape == y.shape == mask.shape == (6686, 10)
    assert len(np.unique(ids)) == len(ids)
    assert mask.all() and np.isfinite(x).all() and np.isfinite(y).all()
    assert (x >= 0).all() and (y >= 0).all()
    return ids, x, y, p

def fit(case, x, y):
    if case == "identity":
        return {"case": case}
    global_scale = max(0., float(np.sum(x*y)/np.sum(x*x)))
    if case == "scale":
        return {"case": case, "global_scale": global_scale}
    if case.startswith("per_state_"):
        alpha = int(case[-3:])/100.
        state_scale = np.maximum(0., np.sum(x*y,axis=0)/np.sum(x*x,axis=0))
        coeff = (1-alpha)*global_scale + alpha*state_scale
        return {"case": case, "global_scale": global_scale,
                "alpha": alpha, "state_scale": state_scale.tolist(),
                "coefficient": coeff.tolist()}
    assert case == "affine_clip"
    design = np.c_[x.reshape(-1), np.ones(x.size)]
    b = np.linalg.lstsq(design, y.reshape(-1), rcond=None)[0]
    return {"case": case, "slope": max(0.,float(b[0])),
            "intercept": float(b[1])}

def predict(model, x):
    case = model["case"]
    if case == "identity":
        return x.copy()
    if case == "scale":
        return x*model["global_scale"]
    if case.startswith("per_state_"):
        return x*np.asarray(model["coefficient"])[None,:]
    return np.maximum(0., x*model["slope"]+model["intercept"])

def metrics(y,p):
    e = y-p
    return {"count": int(y.size), "sse": float(np.sum(e*e)),
            "sst": float(np.sum((y-y.mean())**2)),
            "r2": float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e*e)))}

def diagnostics(y, pred, baseline, bright):
    result = {"pooled": metrics(y,pred),
              "per_state": [metrics(y[:,j],pred[:,j]) for j in range(10)]}
    result["bright_sse"] = {}
    for label, threshold in bright.items():
        m = y>=threshold
        result["bright_sse"][label] = {
            "threshold": threshold, "count": int(m.sum()),
            "sse": float(np.sum((y[m]-pred[m])**2)),
            "baseline_sse": float(np.sum((y[m]-baseline[m])**2)),
            "delta_sse": float(np.sum((y[m]-baseline[m])**2) -
                               np.sum((y[m]-pred[m])**2))}
    return result

def fold_id(ids):
    return np.array([int.from_bytes(hashlib.sha256(str(int(i)).encode()).digest()[:8],
                                    "big")%5 for i in ids], dtype=np.int8)

def crossfit():
    ids,x,y,p=load("val")
    fold=fold_id(ids)
    bright={"top10pct_val": float(np.quantile(y,.9)),
            "top1pct_val": float(np.quantile(y,.99))}
    candidates={}
    for case in CASES:
        oof=np.empty_like(x)
        fold_models=[]
        for k in range(5):
            train=fold!=k
            model=fit(case,x[train],y[train])
            oof[~train]=predict(model,x[~train])
            fold_models.append({"fold":k,"train_molecules":int(train.sum()),
                                "assess_molecules":int((~train).sum()),
                                "model":model})
        candidates[case]={"diagnostics":diagnostics(y,oof,x,bright),
                          "fold_models":fold_models}
    chosen=min(CASES,key=lambda c:(candidates[c]["diagnostics"]["pooled"]["sse"],
                                   CASES.index(c)))
    full_model=fit(chosen,x,y)
    report={"objective":"Improve pooled native oscillator-strength R2 using a small validation-fitted correction to the completed eta0 model.",
            "classification":"Exploratory: this split's test results were analyzed in earlier campaigns and an initial read-only calibration calculation saw test outcomes before this protocol.",
            "prior_test_exposure":{"baseline_test_r2":0.45581510931020497,
                "initial_global_scale_0.89623909_test_r2":0.4585413043325922,
                "initial_global_affine_0.85118302_plus_0.00372519_test_r2":0.4596672330566919,
                "initial_unregularized_per_state_affine_test_r2":0.42597508168998877},
            "source_validation_predictions":str(p),"source_validation_sha256":sha(p),
            "script_sha256":sha(Path(__file__)),
            "fold_rule":"SHA256 of decimal molecule ID, first 8 digest bytes as big-endian integer modulo 5; all 10 states of each molecule stay together.",
            "candidates":candidates,
            "selection_rule":"Lowest pooled out-of-fold native-f SSE; fixed candidate order breaks ties; no test metric used at this stage.",
            "chosen":chosen,"full_validation_fit":full_model,
            "full_validation_fit_metrics":diagnostics(y,predict(full_model,x),x,bright),
            "bright_thresholds_from_validation":bright,
            "test_status":"unopened_by_this_stage"}
    (ROOT/"validation_crossfit.json").write_text(json.dumps(report,indent=2))
    freeze={"chosen":chosen,"model":full_model,
            "validation_report_sha256":sha(ROOT/"validation_crossfit.json"),
            "script_sha256":sha(Path(__file__)),
            "source_val_sha256":sha(p),
            "test_not_opened_during_crossfit_stage":True}
    (ROOT/"frozen_choice.json").write_text(json.dumps(freeze,indent=2))
    print(json.dumps({"selected":chosen,"oof":candidates[chosen]["diagnostics"]["pooled"],
                      "identity_oof":candidates["identity"]["diagnostics"]["pooled"],
                      "full_model":full_model,
                      "validation_report_sha256":freeze["validation_report_sha256"]}))

def test():
    freeze=json.loads((ROOT/"frozen_choice.json").read_text())
    assert freeze["script_sha256"]==sha(Path(__file__))
    assert freeze["validation_report_sha256"]==sha(ROOT/"validation_crossfit.json")
    ids,x,y,p=load("test")
    pred=predict(freeze["model"],x)
    val=json.loads((ROOT/"validation_crossfit.json").read_text())
    bright=val["bright_thresholds_from_validation"]
    b=diagnostics(y,x,x,bright)
    d=diagnostics(y,pred,x,bright)
    rng=np.random.default_rng(20260928)
    diffs=[]
    for _ in range(2000):
        idx=rng.integers(0,len(ids),len(ids))
        yy=y[idx]; xx=x[idx]; pp=pred[idx]
        sst=np.sum((yy-yy.mean())**2)
        diffs.append(float((np.sum((yy-xx)**2)-np.sum((yy-pp)**2))/sst))
    report={"classification":"Single exploratory test evaluation after frozen validation selection; previous historical and preliminary test exposure disclosed in validation_crossfit.json.",
            "frozen_choice_sha256":sha(ROOT/"frozen_choice.json"),
            "test_predictions":str(p),"test_predictions_sha256":sha(p),
            "native_f_truth":"Original printed oscillator_strength, 1e-4 grid, mask_f_true",
            "native_f_prediction":"Saved eta0 model f=(2/3)*(predicted E_eV/27.211386245988)*tr(predicted A); calibration applies to this f.",
            "molecules":len(ids),"states":10,"baseline":b,"calibrated":d,
            "delta_test_r2":d["pooled"]["r2"]-b["pooled"]["r2"],
            "bootstrap":{"resampling_unit":"molecule with all ten states",
                "replicates":2000,"seed":20260928,
                "paired_delta_r2_percentiles_2.5_50_97.5":
                  np.quantile(diffs,[.025,.5,.975]).tolist()},
            "parameters":freeze["model"],
            "further_test_tuning_performed":False}
    (ROOT/"exploratory_test.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"chosen":freeze["chosen"],
                      "baseline":b["pooled"],"calibrated":d["pooled"],
                      "bootstrap":report["bootstrap"],
                      "test_report_sha256":sha(ROOT/"exploratory_test.json")}))

if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in ("crossfit","test")
    globals()[sys.argv[1]]()

