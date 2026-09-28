#!/usr/bin/env python3
"""Validation-only report for completed frozen-head study; no test access."""
import json,sys,time
from pathlib import Path
import numpy as np
from common import ROOT,ARCH,SRC,sha,atomic_json,metric
def main():
    run=ROOT/"run"
    fit=json.loads((run/"FIT_COMPLETE.json").read_text())
    assert fit["event"]=="FIT_COMPLETE" and fit["epochs"]==20 and fit["steps"]==37620
    history=[json.loads(s) for s in (run/"history.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in history]==list(range(21))
    best=fit["best_epoch"]
    assert best==min(range(21),key=lambda j:history[j]["validation"]["raw_f"]["sse"])
    assert sha(run/"best_native_f_val.npz")==sha(run/"selections"/f"epoch{best:03d}_val.npz")
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        ids=z["ids"].copy();indices=z["indices"].copy();y=z["f_true"].copy()
        ctrl=z["f"].copy();Etrue=z["E_true"].copy();Ebase=z["E"].copy()
    arrays={}
    for name,path in (("initial",run/"initial_val.npz"),
                      ("selected",run/"best_native_f_val.npz"),
                      ("final20",run/"final20_val.npz")):
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z["ids"],ids) and np.array_equal(z["indices"],indices)
            assert np.array_equal(z["f_true"],y) and np.array_equal(z["E_true"],Etrue)
            assert np.allclose(z["E_pred"],Ebase,rtol=1e-5,atol=1e-5)
            arrays[name]={k:z[k].copy() for k in z.files}
        m=metric(y,arrays[name]["f_pred"])
        epoch={"initial":0,"selected":best,"final20":20}[name]
        assert abs(m["r2"]-history[epoch]["validation"]["raw_f"]["r2"])<1e-12
    assert np.array_equal(arrays["initial"]["base_f_native64"],
                          arrays["final20"]["base_f_native64"])
    assert np.array_equal(arrays["initial"]["E_pred"],arrays["final20"]["E_pred"])
    assert np.array_equal(arrays["initial"]["base_f32"],arrays["final20"]["base_f32"])
    audit=json.loads((SRC/"data/identity_audit_v2.json").read_text())
    groups=[]
    for index,id_ in zip(indices,ids):
        row=audit[int(index)]
        assert int(row[0])==int(id_) and row[2] is True
        groups.append(row[1])
    sys.path.insert(0,str(ROOT.parent/"analysis_addendum"))
    from validation_comparison_addendum import bootstraps
    bootstrap,layout=bootstraps(y,ctrl,
        {"selected":arrays["selected"]["f_pred"],
         "final20":arrays["final20"]["f_pred"]},groups)
    cfg=json.loads((ROOT/"config.json").read_text())
    control=metric(y,ctrl)
    choices={}
    for name in ("initial","selected","final20"):
        m=metric(y,arrays[name]["f_pred"])
        choices[name]={"epoch":{"initial":0,"selected":best,"final20":20}[name],
            "raw_native_f":m,"delta_r2_vs_eta0":m["r2"]-control["r2"],
            "relative_sse_reduction_vs_eta0":(control["sse"]-m["sse"])/control["sse"],
            "energy_mae":history[{"initial":0,"selected":best,"final20":20}[name]]
                ["validation"]["energy"]["mae"]}
    assert abs(choices["initial"]["raw_native_f"]["r2"]-control["r2"])<5e-7
    prior=json.loads((ARCH/"ARCHITECTURE_RESULTS.json").read_text())["arms"]["retained_residual"]
    report={"classification":"Exploratory validation-only frozen eta0 residual-head comparison; one seed",
        "created":time.time(),"source_fit_sha256":sha(run/"FIT_COMPLETE.json"),
        "source_history_sha256":sha(run/"history.jsonl"),
        "source_prediction_hashes":{name:sha(path) for name,path in
            (("initial",run/"initial_val.npz"),("selected",run/"best_native_f_val.npz"),
             ("final20",run/"final20_val.npz"))},
        "eta0_val_export_sha256":sha(SRC/"runs/mto_eta0/val_predictions.npz"),
        "identity_audit_sha256":sha(SRC/"data/identity_audit_v2.json"),
        "selected_epoch":best,"eta0_control":control,"frozen_head":choices,
        "full_train_f_metrics_by_epoch":[{"epoch":h["epoch"],**h["train"]["raw_f"]}
                                         for h in history],
        "full_validation_f_metrics_by_epoch":[{"epoch":h["epoch"],**h["validation"]["raw_f"]}
                                              for h in history],
        "val_diagnostics_selected":history[best]["validation"],
        "val_diagnostics_final20":history[20]["validation"],
        "paired_bootstrap_vs_eta0":bootstrap,"connectivity_layout":layout,
        "base_and_energy_exactly_invariant_across_initial_final20":True,
        "joint_retained_residual_prior":{"selected_epoch":prior["best_native_f_epoch"],
            "selected_raw_native_f_r2":prior["validation_raw_native_f_r2"],
            "final20_raw_native_f_r2":json.loads((ARCH/"runs/retained_residual/history.jsonl").read_text().splitlines()[-1])["validation"]["raw_f"]["r2"]},
        "limitations":["All metrics use previously reused validation, not an untouched test.",
            "Bootstrap intervals condition on an already selected epoch and one training seed.",
            "Frozen training changes both backbone updates and head gradient clipping; it does not isolate one unique mechanism.",
            "No test, extra epoch, alternate hyperparameter, or model promotion is authorized here."]}
    atomic_json(report,ROOT/"FROZEN_RESIDUAL_RESULTS.json")
    rows=["# Frozen eta0 residual-head validation result","",
          "One reviewed 20-epoch, seed-11 head-only run; source eta0 features, energy and tensor base stayed frozen.",
          "","| Predictor | Epoch | Raw native-f R² | ΔR² vs eta0 | f RMSE | E MAE |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    rows.append(f"| eta0 control | 33 source | {control['r2']:.6f} | 0 | {control['rmse']:.6f} | {history[0]['validation']['energy']['mae']:.6f} |")
    for name in ("initial","selected","final20"):
        v=choices[name];m=v["raw_native_f"]
        rows.append(f"| frozen head {name} | {v['epoch']} | {m['r2']:.6f} | {v['delta_r2_vs_eta0']:+.6f} | {m['rmse']:.6f} | {v['energy_mae']:.6f} |")
    rows.extend(["","Only the validation-selected epoch is a candidate. The prior jointly trained residual selected epoch0 and ended epoch20 at R² "+
        f"{report['joint_retained_residual_prior']['final20_raw_native_f_r2']:.6f}.",
        "Molecule and connectivity-group bootstrap intervals and per-state/tail metrics are in FROZEN_RESIDUAL_RESULTS.json.",
        "No test predictions were opened."])
    (ROOT/"FROZEN_RESIDUAL_RESULTS.md").write_text("\n".join(rows)+"\n")
    print(json.dumps({"complete":True,"selected_epoch":best,
        "selected_r2":choices["selected"]["raw_native_f"]["r2"],
        "delta_vs_eta0":choices["selected"]["delta_r2_vs_eta0"],
        "summary":str(ROOT/"FROZEN_RESIDUAL_RESULTS.json")}))
if __name__=="__main__":main()

