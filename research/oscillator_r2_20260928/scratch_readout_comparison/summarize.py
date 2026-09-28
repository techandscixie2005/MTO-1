#!/usr/bin/env python3
"""Saved-validation-only paired scratch comparison; no model inference or test."""
import json,math,time
from pathlib import Path
import numpy as np
from common import ROOT,SRC,source_seal,code_hashes,sha,save_json,metric
def bootstrap(y,a,b,groups,seed=20260928,reps=2000):
    rng=np.random.default_rng(seed);units=np.unique(groups)
    members=[np.flatnonzero(groups==u) for u in units]
    delta=np.empty(reps)
    for k in range(reps):
        selected=rng.integers(0,len(units),size=len(units))
        rows=np.concatenate([members[j] for j in selected])
        truth=y[rows];sst=np.square(truth-truth.mean()).sum()
        delta[k]=(np.square(b[rows]-truth).sum()-np.square(a[rows]-truth).sum())/sst
    return {"draws":reps,"mean":float(delta.mean()),
        "ci95":[float(x) for x in np.quantile(delta,[.025,.975])],
        "positive_fraction":float((delta>0).mean())}
def main():
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text());review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"]
    assert pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    records={};arrays={}
    for arm in ("native","mto"):
        out=ROOT/"runs"/arm;fit=json.loads((out/"FIT_COMPLETE.json").read_text())
        assert fit["event"]=="FIT_COMPLETE" and fit["epochs"]==100 and fit["steps"]==188100
        assert fit["source_hashes"]==source and fit["code_hashes"]==code
        assert sha(out/"FIXED_CHECKPOINT_METRICS.json")==fit["fixed_metrics_sha256"]
        with np.load(out/"val_best_raw_f.npz",allow_pickle=False) as z:
            arrays[arm]={k:z[k].copy() for k in z.files}
        assert sha(out/"val_best_raw_f.npz")==fit["prediction_hashes"]["best_raw_f"]
        y=arrays[arm]["f_true"];p=arrays[arm]["f"]
        assert abs(metric(y,p)["sse"]-fit["best"]["raw_f"]["metric"])<1e-8
        fixed=json.loads((out/"FIXED_CHECKPOINT_METRICS.json").read_text())
        assert fixed["raw_f"]["epoch"]==fit["best"]["raw_f"]["epoch"]
        records[arm]={"fit_complete_sha256":sha(out/"FIT_COMPLETE.json"),
            "best_epoch":fit["best"]["raw_f"]["epoch"],"best_raw_f":metric(y,p),
            "best_energy":fixed["raw_f"]["val"]["energy"],
            "full_train_initial":fixed["initial"]["train"],
            "full_train_selected":fixed["raw_f"]["train"],
            "full_train_final":fixed["final100"]["train"],
            "val_initial":fixed["initial"]["val"],"val_final":fixed["final100"]["val"],
            "per_state":fixed["raw_f"]["val"]["per_state"],
            "tail":fixed["raw_f"]["val"]["tail"],
            "concentration":fixed["raw_f"]["val"]["molecule_error_concentration"],
            "fixed_case14562":fixed["raw_f"]["val"]["fixed_prior_case14562"]}
    a=arrays["native"];b=arrays["mto"]
    for k in ("ids","indices","f_true","E_true"):
        assert np.array_equal(a[k],b[k]),k
    contextual_predictions=[]
    for arm in ("mto_eta0","mto_eta01","mto_eta1"):
        with np.load(SRC/"runs"/arm/"val_predictions.npz",allow_pickle=False) as z:
            for k in ("ids","indices","f_true","E_true"):
                assert np.array_equal(a[k],z[k]),(arm,k)
            contextual_predictions.append(z["f"].astype(np.float64))
    context={"eta0":metric(a["f_true"],contextual_predictions[0]),
        "fixed_equal_three":metric(a["f_true"],np.mean(contextual_predictions,axis=0))}
    ids=a["ids"];indices=a["indices"];y=a["f_true"]
    table=json.loads((SRC/"data/identity_audit_v2.json").read_text())
    assert len(table)>int(indices.max())
    group=np.array([table[i][1] for i in indices])
    assert np.array_equal(ids,np.array([table[i][0] for i in indices]))
    native=a["f"];mto=b["f"]
    delta=metric(y,native)["r2"]-metric(y,mto)["r2"]
    mol=bootstrap(y,native,mto,np.arange(len(y)))
    clustered=bootstrap(y,native,mto,group)
    result={"classification":"Completed, validation-only paired scratch comparison; selected checkpoints evaluated after selection",
        "primary_difference_native_minus_mto_r2":delta,
        "positive_direction":"native direct readout better",
        "arms":records,"contextual_references":context,
        "selected_native_minus_eta0_r2":records["native"]["best_raw_f"]["r2"]-context["eta0"]["r2"],
        "selected_mto_minus_eta0_r2":records["mto"]["best_raw_f"]["r2"]-context["eta0"]["r2"],
        "bootstrap_molecule":mol,"bootstrap_connectivity_group":clustered,
        "connectivity_groups":int(len(np.unique(group))),
        "source_hashes":source,"code_hashes":code,"test_batches":0,
        "limitations":["One seed11 paired readout-family comparison; architectures differ in routing, capacity and gradient paths.",
            "Validation reused for checkpoint selection and historical research; bootstrap intervals are descriptive and selection conditional.",
            "Equal updates/examples do not imply equal compute or parameter count."],
        "time":time.time()}
    assert abs(mol["mean"]-delta)<.05
    save_json(result,ROOT/"RESULTS.json")
    text=["# Matched scratch native versus MTO direct-f validation comparison","",
        "Both arms used 100 epochs and the same fresh core, train order and normalized raw-f objective.","",
        "| Arm | Selected epoch | Validation raw-f R2 | Train raw-f R2 selected | Final100 validation R2 |",
        "|---|---:|---:|---:|---:|"]
    for arm in ("native","mto"):
        r=records[arm]
        text.append(f"| {arm} | {r['best_epoch']} | {r['best_raw_f']['r2']:.6f} | {r['full_train_selected']['raw_f']['r2']:.6f} | {r['val_final']['raw_f']['r2']:.6f} |")
    text.extend(["",f"Contextual eta0 R2: {context['eta0']['r2']:.6f}; fixed equal-three R2: {context['fixed_equal_three']['r2']:.6f}.",
        f"Native minus MTO selected validation R2: {delta:+.6f}.",
        f"Molecule bootstrap 95% CI: {mol['ci95']}; positive fraction {mol['positive_fraction']:.3f}.",
        f"Connectivity-group bootstrap 95% CI: {clustered['ci95']}; {result['connectivity_groups']} groups.",
        "This is exploratory validation evidence; no test evaluation or promotion is implied.",""])
    (ROOT/"RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    print(json.dumps({"native":records["native"]["best_raw_f"]["r2"],
        "mto":records["mto"]["best_raw_f"]["r2"],"delta":delta}),flush=True)
if __name__=="__main__":main()

