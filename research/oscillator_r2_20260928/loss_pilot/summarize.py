#!/usr/bin/env python3
"""Validation-only matched-arm summary; never opens test split."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"pilot_config.json").read_text())
runs={}
for arm in cfg["arms"]:
    out=ROOT/"runs"/arm
    fit=json.loads((out/"FIT_COMPLETE.json").read_text())
    history=[json.loads(line) for line in (out/"history.jsonl").read_text().splitlines()]
    assert len(history)==cfg["epochs"]+1 and [h["epoch"] for h in history]==list(range(cfg["epochs"]+1))
    best=fit["best_epoch"]["f"]
    ck=json.loads(json.dumps(history[best]["validation"]))
    with np.load(out/"best_f_val.npz") as z:
        arrays={k:z[k].copy() for k in ("ids","indices","f_pred","f_true","E_pred","E_true","f_derived_truth")}
    assert np.array_equal(arrays["indices"],np.load("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data/dataset.npz")["val"])
    yy=arrays["f_true"];pp=arrays["f_pred"]
    derived_r2=1-np.square(yy-pp).sum()/np.square(yy-yy.mean()).sum()
    assert abs(derived_r2-ck["raw_f"]["r2"])<1e-8
    runs[arm]={"fit":fit,"history":history,"best_f_epoch":best,"best_f_validation":ck,"arrays":arrays}
truth=runs["control"]["arrays"]["f_true"]
ids=runs["control"]["arrays"]["ids"]
assert truth.shape==(6686,10)
for arm,r in runs.items():
    assert np.array_equal(r["arrays"]["ids"],ids)
    assert np.array_equal(r["arrays"]["f_true"],truth)
rng=np.random.default_rng(cfg["bootstrap_seed"])
bootstrap_indices=rng.integers(0,len(ids),(cfg["bootstrap_samples"],len(ids)))
sst=np.square(truth-truth.mean()).sum()
def info(arm):
    a=runs[arm]["arrays"];pred=a["f_pred"];err=np.square(pred-truth)
    sse=float(err.sum());baseline=runs["control"]["arrays"]["f_pred"]
    delta=float((np.square(baseline-truth)-err).sum()/sst)
    control_sse=float(np.square(baseline-truth).sum())
    paired=[]
    for ix in bootstrap_indices:
        yy=truth[ix];pc=baseline[ix];pp=pred[ix]
        paired.append(float((np.square(pc-yy).sum()-np.square(pp-yy).sum())/
                            np.square(yy-yy.mean()).sum()))
    paired=np.asarray(paired)
    bright={}
    for q in (.9,.99):
        th=float(np.quantile(truth,q));m=truth>=th
        bright[str(q)]={"threshold":th,"count":int(m.sum()),
            "sse":float(err[m].sum()),
            "control_sse":float(np.square(baseline[m]-truth[m]).sum())}
    epoch0_mae=runs[arm]["history"][0]["validation"]["energy"]["mae"]
    best_mae=runs[arm]["best_f_validation"]["energy"]["mae"]
    return {"best_f_epoch":runs[arm]["best_f_epoch"],"best_f_val_r2":1-sse/sst,
            "best_f_val_sse":sse,"delta_r2_vs_control":delta,
            "relative_sse_reduction_vs_control":(control_sse-sse)/control_sse,
            "derived_f_truth_r2":runs[arm]["best_f_validation"]["derived_f"]["r2"],
            "bootstrap_delta_r2_2p5_50_97p5":np.quantile(paired,[.025,.5,.975]).tolist(),
            "bootstrap_delta_r2_positive_fraction":float((paired>0).mean()),
            "energy_mae_best":best_mae,"energy_mae_epoch0":epoch0_mae,
            "energy_mae_ratio_to_epoch0":best_mae/epoch0_mae,
            "energy_drift_review_flag":best_mae/epoch0_mae>cfg["promotion"]["energy_MAE_ratio_review_flag"],
            "per_state_r2":[m["r2"] for m in runs[arm]["best_f_validation"]["raw_f"]["per_state"]],
            "bright":bright,
            "best_old_objective_epoch":runs[arm]["fit"]["best_epoch"]["old"],
            "best_arm_objective_epoch":runs[arm]["fit"]["best_epoch"]["arm"],
            "best_old_objective_val_r2":runs[arm]["history"][runs[arm]["fit"]["best_epoch"]["old"]]["validation"]["raw_f"]["r2"],
            "best_arm_objective_val_r2":runs[arm]["history"][runs[arm]["fit"]["best_epoch"]["arm"]]["validation"]["raw_f"]["r2"]}
out={"classification":"Matched continuation screen, validation only; no test metrics","source":"completed eta0 epoch33",
    "selection":"lowest pooled raw native-f validation SSE among epochs 0..20 per arm",
    "arms":{arm:info(arm) for arm in cfg["arms"]},
    "promotion_rule":cfg["promotion"]}
(ROOT/"pilot_summary.json").write_text(json.dumps(out,indent=2))
lines=["# Matched oscillator-strength continuation screen","","All results below use the frozen validation split and the original raw oscillator-strength label. No test split was opened.","",
       "| Arm | Best epoch | Native-f val R² | ΔR² vs control | Bootstrap 95% CI | Energy MAE ratio |",
       "| --- | ---: | ---: | ---: | --- | ---: |"]
for arm,r in out["arms"].items():
    ci=r["bootstrap_delta_r2_2p5_50_97p5"]
    lines.append(f"| {arm} | {r['best_f_epoch']} | {r['best_f_val_r2']:.6f} | {r['delta_r2_vs_control']:+.6f} | [{ci[0]:+.6f}, {ci[2]:+.6f}] | {r['energy_mae_ratio_to_epoch0']:.3f} |")
lines += ["","## Interpretation","The raw metrics, per-state scores, bright-state SSE, and checkpoint-selection comparisons are in pilot_summary.json. The research coordinator should decide whether any arm merits replication; this script makes no test-set or promotion decision."]
(ROOT/"PILOT_SUMMARY.md").write_text("\n".join(lines)+"\n")
print(json.dumps({a:{"r2":v["best_f_val_r2"],"delta":v["delta_r2_vs_control"]} for a,v in out["arms"].items()}))

