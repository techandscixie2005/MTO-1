#!/usr/bin/env python3
"""Completed frozen-head validation benchmark versus eta0 and fixed equal-three ensemble."""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=pathlib.Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    fit=json.loads((ROOT/"run/FIT_COMPLETE.json").read_text())
    assert fit["event"]=="FIT_COMPLETE"
    preds={};ids=None;indices=None;y=None;hashes={}
    for name in ("mto_eta0","mto_eta01","mto_eta1"):
        p=SRC/"runs"/name/"val_predictions.npz";hashes[name]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            if ids is None:ids=z["ids"].copy();indices=z["indices"].copy();y=z["f_true"].copy()
            else:
                assert np.array_equal(z["ids"],ids)
                assert np.array_equal(z["indices"],indices)
                assert np.array_equal(z["f_true"],y)
            preds[name]=z["f"].copy()
    equal3=sum(preds.values())/3
    with np.load(ROOT/"run/best_native_f_val.npz",allow_pickle=False) as z:
        assert np.array_equal(z["ids"],ids) and np.array_equal(z["indices"],indices)
        assert np.array_equal(z["f_true"],y)
        selected=z["f_pred"].copy()
    audit=json.loads((SRC/"data/identity_audit_v2.json").read_text())
    groups=[]
    for index,id_ in zip(indices,ids):
        row=audit[int(index)]
        assert int(row[0])==int(id_) and row[2] is True
        groups.append(row[1])
    sys.path.insert(0,str(ROOT.parent/"analysis_addendum"))
    from validation_comparison_addendum import bootstraps,metric
    b0,layout=bootstraps(y,preds["mto_eta0"],{"frozen_selected":selected},groups)
    b3,layout2=bootstraps(y,equal3,{"frozen_selected":selected},groups)
    assert layout==layout2
    report={"classification":"Validation-only context comparison; selected frozen single model versus pre-existing fixed equal-three ensemble",
        "created":time.time(),"source_fit_sha256":sha(ROOT/"run/FIT_COMPLETE.json"),
        "source_prediction_hashes":hashes,
        "selected_prediction_sha256":sha(ROOT/"run/best_native_f_val.npz"),
        "selected_epoch":fit["best_epoch"],"metrics":{"eta0":metric(y,preds["mto_eta0"]),
            "equal_three_native_f_mean":metric(y,equal3),"frozen_selected":metric(y,selected)},
        "paired_delta_r2_frozen_minus_eta0":{"point":metric(y,selected)["r2"]-metric(y,preds["mto_eta0"])["r2"],
            "bootstrap":b0},
        "paired_delta_r2_frozen_minus_equal_three":{"point":metric(y,selected)["r2"]-metric(y,equal3)["r2"],
            "bootstrap":b3},
        "connectivity_layout":layout,
        "limitations":["Fixed equal-three average uses each constituent model's native f prediction; E/A are not averaged.",
            "All intervals condition on validation-selected checkpoints and a reused validation split.",
            "One seed; no test access or model promotion."]}
    p=ROOT/"FROZEN_VS_ENSEMBLE_VALIDATION.json"
    p.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"written":str(p),"selected_r2":report["metrics"]["frozen_selected"]["r2"],
        "equal_three_r2":report["metrics"]["equal_three_native_f_mean"]["r2"]}))
if __name__=="__main__":main()

