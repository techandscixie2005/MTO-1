#!/usr/bin/env python3
"""Validation-only architecture comparison after all three arms finish."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"config.json").read_text())
arms={}
with np.load("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data/dataset.npz") as z:
    val_indices=z["val"].copy();val_ids=z["ids"][val_indices].copy()
for arm in cfg["arms"]:
    out=ROOT/"runs"/arm
    fit=json.loads((out/"FIT_COMPLETE.json").read_text())
    hist=[json.loads(s) for s in (out/"history.jsonl").read_text().splitlines()]
    assert len(hist)==cfg["epochs"]+1 and [x["epoch"] for x in hist]==list(range(cfg["epochs"]+1))
    ep=fit["best_epoch"]["f"];best=hist[ep]["validation"]
    with np.load(out/"best_native_f_val.npz",allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ("ids","indices","f_pred","f_true","E_pred","E_true","f_derived_truth")}
    assert np.array_equal(arrays["indices"],val_indices) and np.array_equal(arrays["ids"],val_ids)
    y=arrays["f_true"];p=arrays["f_pred"]
    r2=1-np.square(p-y).sum()/np.square(y-y.mean()).sum()
    assert abs(r2-best["raw_f"]["r2"])<1e-8
    arms[arm]={"fit":fit,"history":hist,"best":best,"arrays":arrays}
truth=arms["original"]["arrays"]["f_true"]
for arm in cfg["arms"]:assert np.array_equal(arms[arm]["arrays"]["f_true"],truth)
base=arms["original"]["arrays"]["f_pred"]
sst=float(np.square(truth-truth.mean()).sum())
base_sse=float(np.square(base-truth).sum())
rng=np.random.default_rng(cfg["bootstrap_seed"])
ixs=rng.integers(0,len(truth),(cfg["bootstrap_samples"],len(truth)))
results={}
for arm in cfg["arms"]:
    r=arms[arm];p=r["arrays"]["f_pred"]
    sse=float(np.square(p-truth).sum())
    boot=[]
    for ix in ixs:
        y=truth[ix];b=base[ix];v=p[ix]
        boot.append(float((np.square(b-y).sum()-np.square(v-y).sum())/
                    np.square(y-y.mean()).sum()))
    boot=np.asarray(boot)
    ep=r["fit"]["best_epoch"]["f"];loss_ep=r["fit"]["best_epoch"]["loss"]
    e0=r["history"][0]["validation"]["energy"]["mae"]
    eb=r["best"]["energy"]["mae"]
    results[arm]={"best_native_f_epoch":ep,"validation_raw_native_f_r2":1-sse/sst,
        "validation_raw_native_f_sse":sse,
        "delta_r2_vs_original":(base_sse-sse)/sst,
        "relative_sse_reduction_vs_original":(base_sse-sse)/base_sse,
        "bootstrap_delta_r2_2p5_50_97p5":np.quantile(boot,[.025,.5,.975]).tolist(),
        "bootstrap_positive_fraction":float((boot>0).mean()),
        "energy_mae_epoch0":e0,"energy_mae_best":eb,"energy_mae_ratio_to_epoch0":eb/e0,
        "energy_drift_review_flag":eb/e0>cfg["promotion"]["energy_MAE_ratio_review_flag"],
        "per_state":r["best"]["per_state"],"bright":r["best"]["bright"],
        "derived_f_r2":r["best"]["derived_f"]["r2"],
        "best_common_objective_epoch":loss_ep,
        "best_common_objective_validation_f_r2":r["history"][loss_ep]["validation"]["raw_f"]["r2"],
        "parameter_counts":json.loads((ROOT/"PREPARATION.json").read_text())["arms"][arm]["parameter_counts"],
        "initialization":json.loads((ROOT/"PREPARATION.json").read_text())["arms"][arm]}
out={"classification":"Validation-only matched architecture screen, 20 epochs, no test evaluation",
    "primary_metric":"pooled raw native-f R2; best epoch chosen by validation f SSE",
    "arms":results,"promotion_rule":cfg["promotion"],
    "limitations":["one seed","checkpoint selection uses validation","new scalar-head arms have extra train-only distillation compute"]}
(ROOT/"ARCHITECTURE_RESULTS.json").write_text(json.dumps(out,indent=2))
lines=["# Matched MTO architecture screen","","All figures use the frozen validation split and original raw oscillator-strength labels. No historical test predictions were opened.","",
       "| Arm | Best epoch | Native-f validation R² | ΔR² versus original | Bootstrap 95% interval | Energy MAE ratio |",
       "| --- | ---: | ---: | ---: | --- | ---: |"]
for arm,v in results.items():
    ci=v["bootstrap_delta_r2_2p5_50_97p5"]
    lines.append(f"| {arm} | {v['best_native_f_epoch']} | {v['validation_raw_native_f_r2']:.6f} | {v['delta_r2_vs_original']:+.6f} | [{ci[0]:+.6f}, {ci[2]:+.6f}] | {v['energy_mae_ratio_to_epoch0']:.3f} |")
lines += ["","Full per-state, bright-transition, energy, initialization and objective-selection metrics are in ARCHITECTURE_RESULTS.json. The research coordinator decides whether any arm merits extension or replication."]
(ROOT/"ARCHITECTURE_RESULTS.md").write_text("\n".join(lines)+"\n")
print(json.dumps({k:{"r2":v["validation_raw_native_f_r2"],"delta":v["delta_r2_vs_original"]} for k,v in results.items()}))

