#!/usr/bin/env python3
"""Post-completion validation-only fixed seed comparisons; no checkpoint choice."""
import json,math,sys,time
from pathlib import Path
import numpy as np
from common import ROOT,SRC,source_seal,code_hashes,sha,save_json,metric
ADD=ROOT.parent/"analysis_addendum"
sys.path.insert(0,str(ADD))
from validation_comparison_addendum import bootstraps
def load(path,ids=None,indices=None,truth=None):
    with np.load(path,allow_pickle=False) as z:
        got={k:z[k].copy() for k in ("ids","indices","f_true","f")}
    if ids is not None:
        assert np.array_equal(got["ids"],ids) and np.array_equal(got["indices"],indices)
        assert np.array_equal(got["f_true"],truth)
    return got
def main():
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"] and pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    seed_files={}
    for seed in (23,37):
        out=ROOT/f"runs/seed{seed}"
        fit=json.loads((out/"FIT_COMPLETE.json").read_text())
        assert fit["event"]=="FIT_COMPLETE" and fit["epochs"]==100 and fit["steps"]==188100
        assert fit["source_hashes"]==source and fit["code_hashes"]==code
        assert sha(out/"FIXED_CHECKPOINT_METRICS.json")==fit["fixed_metrics_sha256"]
        for name,digest in fit["checkpoint_hashes"].items():
            path={"initial":"initial.pt","legacy":"best_legacy.pt","raw_f":"best_raw_f.pt",
                  "final100":"final100.pt","last":"last.pt"}[name]
            assert sha(out/path)==digest
        for name,digest in fit["val_prediction_hashes"].items():
            assert sha(out/f"val_{name}.npz")==digest
        seed_files[seed]=fit
    old=load(SRC/"runs/mto_eta0/val_predictions.npz")
    ids=old["ids"];indices=old["indices"];truth=old["f_true"]
    pred={"seed11_legacy":old["f"]}
    paths={}
    for seed in (23,37):
        out=ROOT/f"runs/seed{seed}"
        for label,name in (("legacy","best_legacy"),("raw_f_secondary","best_raw_f")):
            path=out/f"val_{name}.npz"
            pred[f"seed{seed}_{label}"]=load(path,ids,indices,truth)["f"]
            paths[f"seed{seed}_{label}"]=sha(path)
    ens=[]
    for name in ("mto_eta0","mto_eta01","mto_eta1"):
        path=SRC/"runs"/name/"val_predictions.npz"
        ens.append(load(path,ids,indices,truth)["f"])
    pred["fixed_equal_three"]=sum(ens)/3
    pred["primary_equal_seeds_11_23_37"]=(pred["seed11_legacy"]+
        pred["seed23_legacy"]+pred["seed37_legacy"])/3
    pred["secondary_mixed_selection_equal_three"]=(pred["seed11_legacy"]+
        pred["seed23_raw_f_secondary"]+pred["seed37_raw_f_secondary"])/3
    identity=json.loads((SRC/"data/identity_audit_v2.json").read_text())
    groups=[]
    for ix,id_ in zip(indices,ids):
        row=identity[int(ix)]
        assert int(row[0])==int(id_) and row[2] is True
        groups.append(row[1])
    metrics={name:metric(truth,p) for name,p in pred.items()}
    primary={"primary_equal_seeds_11_23_37":pred["primary_equal_seeds_11_23_37"],
             "secondary_mixed_selection_equal_three":pred["secondary_mixed_selection_equal_three"],
             "seed23_legacy":pred["seed23_legacy"],"seed37_legacy":pred["seed37_legacy"]}
    bs_old,layout=bootstraps(truth,pred["seed11_legacy"],primary,groups)
    bs_cost,layout2=bootstraps(truth,pred["fixed_equal_three"],
        {"primary_equal_seeds_11_23_37":pred["primary_equal_seeds_11_23_37"],
         "secondary_mixed_selection_equal_three":pred["secondary_mixed_selection_equal_three"]},groups)
    assert layout==layout2
    err={name:(p-truth).reshape(-1) for name,p in pred.items()}
    correlation={}
    for name in ("seed23_legacy","seed37_legacy","fixed_equal_three"):
        correlation[name]=float(np.corrcoef(err["seed11_legacy"],err[name])[0,1])
    correlation["seed23_vs_seed37_legacy"]=float(np.corrcoef(
        err["seed23_legacy"],err["seed37_legacy"])[0,1])
    concentration={}
    case14562={}
    case=np.flatnonzero(ids==14562)
    assert len(case)==1
    case=int(case[0])
    for name,p in pred.items():
        per_mol=np.square(p-truth).sum(axis=1)
        order=np.argsort(-per_mol,kind="stable")
        total=float(per_mol.sum())
        concentration[name]={label:{"molecules":k,
            "share_of_sse":float(per_mol[order[:k]].sum()/total)}
            for label,k in (("top1",1),("top10",10),
                ("top1pct",max(1,math.ceil(.01*len(ids)))))}
        case14562[name]={"id":14562,"global_index":int(indices[case]),
            "raw_f":truth[case].tolist(),"prediction":p[case].tolist(),
            "molecule_sse":float(per_mol[case])}
    state={}
    tail={}
    for name in ("primary_equal_seeds_11_23_37","secondary_mixed_selection_equal_three"):
        p=pred[name]
        state[name]=[{"state":j+1,**metric(truth[:,j],p[:,j])} for j in range(10)]
        tail[name]={}
        for q,t in (("q90",.0546),("q99",.2377)):
            m=truth>=t
            tail[name][q]={"threshold":t,"bright_count":int(m.sum()),
                "bright_sse":float(np.square(p[m]-truth[m]).sum()),
                "below_count":int((~m).sum()),
                "below_sse":float(np.square(p[~m]-truth[~m]).sum())}
    result={"classification":"Prespecified two-seed replication, validation-only comparisons",
        "time":time.time(),"source_hashes":source,"code_hashes":code,
        "completion_hashes":{str(seed):sha(ROOT/f"runs/seed{seed}/FIT_COMPLETE.json") for seed in (23,37)},
        "validation_array_hashes":paths,"metrics":metrics,
        "primary":"Three legacy-selected checkpoints: archived seed11 epoch33; fresh seeds23/37 min legacy objective epochs1..100",
        "secondary":"Old seed11 legacy-selected plus fresh seed23/37 raw-f-selected; mixed selection, not all-raw-f-selected",
        "delta_r2":{"primary_minus_seed11":metrics["primary_equal_seeds_11_23_37"]["r2"]-metrics["seed11_legacy"]["r2"],
            "primary_minus_fixed_equal_three":metrics["primary_equal_seeds_11_23_37"]["r2"]-metrics["fixed_equal_three"]["r2"],
            "secondary_minus_seed11":metrics["secondary_mixed_selection_equal_three"]["r2"]-metrics["seed11_legacy"]["r2"]},
        "bootstrap_vs_seed11":bs_old,"bootstrap_vs_fixed_equal_three":bs_cost,
        "connectivity_layout":layout,"error_correlation_with_seed11":correlation,
        "per_state":state,"tail":tail,
        "molecule_error_concentration":concentration,
        "fixed_prior_case14562":case14562,
        "limitations":["Three chosen seeds are a small robustness sample.",
            "Different seeds vary initialization and training order together.",
            "Validation and feature choices are historically exposed; bootstrap intervals are descriptive.",
            "Fresh 100-epoch policies differ in duration from archived seed11 236-epoch run.",
            "No test access, validation-driven extension, subset choice or ensemble weight fitting."]}
    save_json(result,ROOT/"RESULTS.json")
    lines=["# Eta0 seed23/37 replication","","Validation raw native-f results from fixed checkpoint rules.",
        "","| Predictor | R² | ΔR² vs seed11 | RMSE |","| --- | ---: | ---: | ---: |"]
    for name in ("seed11_legacy","seed23_legacy","seed37_legacy",
                 "primary_equal_seeds_11_23_37","secondary_mixed_selection_equal_three","fixed_equal_three"):
        m=metrics[name]
        lines.append(f"| {name} | {m['r2']:.6f} | {m['r2']-metrics['seed11_legacy']['r2']:+.6f} | {m['rmse']:.6f} |")
    lines+=["","Primary averages three legacy-selected seed checkpoints. The secondary average mixes old legacy selection with new raw-f selection.",
            "Full paired molecule/connectivity bootstrap, state and tail diagnostics are in RESULTS.json.",
            "No test access or ensemble-weight fitting."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"primary_r2":metrics["primary_equal_seeds_11_23_37"]["r2"],
        "fixed_equal_three_r2":metrics["fixed_equal_three"]["r2"]}),flush=True)
if __name__=="__main__":main()

