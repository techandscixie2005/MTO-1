#!/usr/bin/env python3
"""CPU-only validation addendum. Reads completed runs or frozen validation fixtures."""
import argparse,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
IDENTITY=SRC/"data/identity_audit_v2.json"
SEED=20260928
REPS=2000
C_F=2/(3*27.211386245988)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()
def save(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(p)
def frozen_data():
    with np.load(SRC/"data/dataset.npz",allow_pickle=False) as z:
        ix=z["val"].copy();ids=z["ids"][ix].copy()
        assert len(z["train"])==120355 and len(ix)==6686
        assert not np.intersect1d(z["train"],ix).size
    with np.load(SRC/"data/raw_labels.npz",allow_pickle=False) as z:
        assert np.array_equal(z["ids"][ix],ids)
        y=z["f"][ix].astype(np.float64)
        e=z["E"][ix].astype(np.float64)
        a=z["A"][ix].astype(np.float64)
        for name in ("mask_E","mask_A","mask_f"):assert z[name][ix].all()
    assert y.shape==(6686,10) and np.isfinite(y).all()
    assert len(set(ids.tolist()))==len(ids) and len(set(ix.tolist()))==len(ix)
    audit=json.loads(IDENTITY.read_text())
    assert len(audit)>int(ix.max())
    keys=[]
    for row,id_ in zip(ix,ids):
        item=audit[int(row)]
        assert int(item[0])==int(id_) and item[2] is True
        keys.append(item[1])
    return {"indices":ix,"ids":ids,"y":y,"E":e,
            "derived":C_F*e*np.trace(a,axis1=-2,axis2=-1),
            "groups":keys,
            "hashes":{str(p):sha(p) for p in (SRC/"data/dataset.npz",
                SRC/"data/raw_labels.npz",IDENTITY)}}
def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    err=p-y;sst=float(np.square(y-y.mean()).sum())
    assert sst>0
    return {"count":int(y.size),"sse":float(np.square(err).sum()),
            "sst":sst,"r2":float(1-np.square(err).sum()/sst),
            "mae":float(np.abs(err).mean()),
            "rmse":float(np.sqrt(np.square(err).mean()))}
def load_arrays(path,data,kind="f_pred"):
    with np.load(path,allow_pickle=False) as z:
        expected=("ids","indices",kind,"f_true")
        assert set(expected).issubset(z.files)
        ids=z["ids"].copy();ix=z["indices"].copy()
        pred=z[kind].astype(np.float64);truth=z["f_true"].astype(np.float64)
        extra={k:z[k].astype(np.float64) for k in ("E_pred","E_true","f_derived_truth")
               if k in z.files}
    assert np.array_equal(ids,data["ids"]) and np.array_equal(ix,data["indices"])
    assert np.array_equal(truth,data["y"]) and pred.shape==truth.shape==(6686,10)
    assert np.isfinite(pred).all()
    if "E_true" in extra:assert np.array_equal(extra["E_true"],data["E"])
    if "f_derived_truth" in extra:
        assert np.allclose(extra["f_derived_truth"],data["derived"],rtol=0,atol=1e-12)
    if "E_pred" in extra:assert np.isfinite(extra["E_pred"]).all()
    return pred,extra
def cluster_layout(keys):
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[key].append(i)
    units=list(groups.values())
    return units,{"molecules":len(keys),"groups":len(units),
        "groups_with_multiple_molecules":sum(len(u)>1 for u in units),
        "largest_group_molecules":max(len(u) for u in units)}
def bootstraps(y,base,preds,groups):
    assert all(p.shape==y.shape for p in preds.values())
    names=list(preds)
    molecule=[[i] for i in range(len(y))]
    clusters,layout=cluster_layout(groups)
    outcome={}
    for unit_name,units in (("molecule",molecule),("connectivity_group",clusters)):
        n=np.array([y[u].size for u in units],dtype=np.float64)
        sy=np.array([y[u].sum() for u in units],dtype=np.float64)
        sy2=np.array([np.square(y[u]).sum() for u in units],dtype=np.float64)
        be=np.square(base-y).sum(axis=1)
        changes=np.stack([be-np.square(preds[name]-y).sum(axis=1) for name in names])
        delta=np.array([[v[u].sum() for u in units] for v in changes])
        rng=np.random.default_rng(SEED)
        values=np.empty((len(names),REPS),dtype=np.float64)
        for b in range(REPS):
            draw=rng.integers(0,len(units),len(units))
            sst=sy2[draw].sum()-sy[draw].sum()**2/n[draw].sum()
            assert sst>0
            values[:,b]=delta[:,draw].sum(axis=1)/sst
            if b==0 and unit_name=="molecule":
                yy=y[draw];bb=base[draw]
                direct=[(np.square(bb-yy).sum()-np.square(preds[name][draw]-yy).sum())/
                        np.square(yy-yy.mean()).sum() for name in names]
                assert np.allclose(values[:,b],direct,rtol=0,atol=1e-12)
        outcome[unit_name]={name:{"delta_r2_2p5_50_97p5":
            np.quantile(values[i],[.025,.5,.975]).tolist(),
            "positive_fraction":float((values[i]>0).mean()),
            "replicates":REPS,"seed":SEED,
            "unit":"whole molecule, all 10 states" if unit_name=="molecule" else
                   "connectivity group, all molecules and states"}
            for i,name in enumerate(names)}
    return outcome,layout
def decomposition(data,base,pred):
    y=data["y"];sst=float(np.square(y-y.mean()).sum())
    ctrl=np.square(base-y);err=np.square(pred-y)
    out=[]
    for j in range(10):
        sj=float(err[:,j].sum());bj=float(ctrl[:,j].sum())
        state_sst=float(np.square(y[:,j]-y[:,j].mean()).sum())
        out.append({"state":j+1,"state_index_zero_based":j,
                    "count":len(y),"sse":sj,"control_sse":bj,
                    "delta_sse":bj-sj,
                    "delta_r2_contribution_pooled_sst":(bj-sj)/sst,
                    "state_r2":1-sj/state_sst,
                    "control_state_r2":1-bj/state_sst})
    assert abs(sum(r["delta_r2_contribution_pooled_sst"] for r in out)-
               (ctrl.sum()-err.sum())/sst)<1e-12
    return out
def tail(data,base,pred):
    y=data["y"];ce=np.square(base-y);ae=np.square(pred-y)
    out={}
    for q in (.9,.99):
        threshold=float(np.quantile(y,q))
        subsets={"bright":y>=threshold,"complement":y<threshold}
        cuts={}
        for label,mask in subsets.items():
            b=float(ce[mask].sum());v=float(ae[mask].sum())
            cuts[label]={"count":int(mask.sum()),"control_sse":b,"sse":v,
                         "delta_sse":b-v,
                         "relative_sse_reduction_vs_control":(b-v)/b if b>0 else None}
        assert sum(c["count"] for c in cuts.values())==y.size
        out[str(q)]={"validation_raw_f_threshold":threshold,"subsets":cuts}
    return out
def source_seal(study,fit):
    pre=RESEARCH/("preflight_results.json" if study=="pilot" else "architecture/PREFLIGHT.json")
    report=json.loads(pre.read_text())
    assert report["passed"] and fit["source_hashes"]==report["source_hashes"]
    for path,digest in report["source_hashes"].items():
        assert sha(path)==digest,"Source or data changed: "+path
    return {"preflight_path":str(pre),"preflight_sha256":sha(pre),
            "source_file_count":len(report["source_hashes"]),
            "fit_source_hashes_sha256":hashlib.sha256(json.dumps(
                fit["source_hashes"],sort_keys=True).encode()).hexdigest()}
def study_paths(study):
    assert study in ("pilot","architecture")
    base=RESEARCH if study=="pilot" else RESEARCH/"architecture"
    cfg=json.loads((base/("pilot_config.json" if study=="pilot" else "config.json")).read_text())
    predname="best_f_val.npz" if study=="pilot" else "best_native_f_val.npz"
    summary="pilot_summary.json" if study=="pilot" else "ARCHITECTURE_RESULTS.json"
    required=[base/summary]
    for arm in cfg["arms"]:
        run=base/"runs"/arm
        required += [run/"FIT_COMPLETE.json",run/"history.jsonl",run/predname]
    return base,cfg,predname,required
def ready(study):
    base,cfg,predname,paths=study_paths(study)
    missing=[str(p) for p in paths if not p.exists()]
    return None if not missing else {"study":study,"status":"pending",
        "missing_required_completed_artifacts":missing,
        "note":"No running histories, predictions, or checkpoints were opened."}
def one_study(study,data):
    pending=ready(study)
    if pending:return pending
    base,cfg,predname,_=study_paths(study)
    assert cfg["epochs"]==20 and cfg["allow_test_evaluation"] is False
    control="control" if study=="pilot" else "original"
    summary=json.loads((base/("pilot_summary.json" if study=="pilot" else
                               "ARCHITECTURE_RESULTS.json")).read_text())
    arms={};preds={}
    seal=None
    for arm in cfg["arms"]:
        run=base/"runs"/arm
        fit=json.loads((run/"FIT_COMPLETE.json").read_text())
        assert fit["arm"]==arm and fit["epochs"]==20
        current=source_seal(study,fit)
        if seal is None:seal=current
        else:assert seal==current
        history=[json.loads(v) for v in (run/"history.jsonl").read_text().splitlines()]
        assert len(history)==21 and [v["epoch"] for v in history]==list(range(21))
        selected=int(fit["best_epoch"]["f"])
        assert selected==int(np.argmin([v["validation"]["raw_f"]["sse"] for v in history]))
        checkpoint=run/"selections"/f"f_epoch{selected:03d}.pt"
        selected_pred=run/"selections"/f"f_epoch{selected:03d}_val.npz"
        assert checkpoint.exists() and selected_pred.exists()
        assert sha(selected_pred)==sha(run/predname)
        alias=run/("best_f.pt" if study=="pilot" else "best_native_f.pt")
        assert sha(checkpoint)==sha(alias)
        p,extra=load_arrays(run/predname,data)
        calc=metric(data["y"],p)
        observed=history[selected]["validation"]["raw_f"]
        assert abs(calc["r2"]-observed["r2"])<1e-8
        assert abs(calc["sse"]-observed["sse"])<1e-7
        if "E_pred" in extra:
            assert abs(float(np.abs(extra["E_pred"]-data["E"]).mean())-
                       history[selected]["validation"]["energy"]["mae"])<1e-7
        key="best_f_val_r2" if study=="pilot" else "validation_raw_native_f_r2"
        assert abs(calc["r2"]-summary["arms"][arm][key])<1e-8
        epochs={}
        for label,ep in (("epoch0",0),("selected_best",selected),("final20",20)):
            v=history[ep]["validation"]
            f=v["raw_f"];e=v["energy"]
            epochs[label]={"epoch":ep,"raw_native_f_r2":f["r2"],
                "raw_native_f_sse":f["sse"],"raw_native_f_mae":f["mae"],
                "raw_native_f_rmse":f["rmse"],"energy_mae":e["mae"]}
        curve=[float(v["validation"]["raw_f"]["r2"]) for v in history]
        arms[arm]={"epoch_metrics":epochs,"r2_curve_epochs0_to20":curve,
            "curve_context":{"median_r2":float(np.median(curve)),
                             "best_minus_epoch0":curve[selected]-curve[0],
                             "best_minus_final20":curve[selected]-curve[20],
                             "epochs_within_0p005_of_best":sum(r>=curve[selected]-.005 for r in curve)},
            "selected_checkpoint_sha256":sha(checkpoint),
            "selected_predictions_sha256":sha(run/predname),
            "fit_complete_sha256":sha(run/"FIT_COMPLETE.json"),
            "derived_truth_r2":metric(data["derived"],p)["r2"],
            "initialization_label":("pretrained_eta0_exact" if study=="pilot" or
                arm in ("original","retained_residual") else
                "train_only_teacher_distilled_scalar_head")}
        preds[arm]=p
    basepred=preds[control]
    bootstrap,layout=bootstraps(data["y"],basepred,preds,data["groups"])
    sst=float(np.square(data["y"]-data["y"].mean()).sum())
    for arm in cfg["arms"]:
        p=preds[arm];err=np.square(p-data["y"]);baseerr=np.square(basepred-data["y"])
        arms[arm]["selected_raw_native_f"]=metric(data["y"],p)
        arms[arm]["matched_control"]=control
        arms[arm]["delta_r2_vs_matched_control"]=float((baseerr.sum()-err.sum())/sst)
        arms[arm]["relative_sse_reduction_vs_matched_control"]=float(
            (baseerr.sum()-err.sum())/baseerr.sum())
        arms[arm]["per_state"]=decomposition(data,basepred,p)
        arms[arm]["tail"]=tail(data,basepred,p)
        arms[arm]["bootstrap"]= {k:v[arm] for k,v in bootstrap.items()}
    direct_denom=C_F**2*3*json.loads((RESEARCH/"pilot_config.json").read_text())["sA2"]*json.loads(
        (RESEARCH/"pilot_config.json").read_text())["mean_E2_train"]
    arch_denom=json.loads((RESEARCH/"architecture/stats.json").read_text())["f_var_train"]
    assert abs(direct_denom-0.011321718689897071)<1e-12
    assert abs(arch_denom-0.002510981243894732)<1e-12
    ratio=direct_denom/arch_denom
    assert abs(ratio-4.508882221810699)<1e-9
    result={"status":"complete","study":study,"classification":"validation-only selected single models",
        "created":time.time(),"matched_control":control,"arms":arms,
        "connectivity_layout":layout,"source_seal":seal,
        "raw_label_data_hashes":data["hashes"],
        "common_validation_thresholds":{"q90":float(np.quantile(data["y"],.9)),
            "q99":float(np.quantile(data["y"],.99))},
        "objective_scale":{"pilot_direct_f_matched_denominator":direct_denom,
            "architecture_f_variance_denominator":arch_denom,
            "architecture_f_weight_relative_to_pilot_direct_f_matched":ratio,
            "pilot_control_and_weighted_use_trace_losses":True},
        "limitations":["The validation split selected the best checkpoint in each arm and informed arm design; bootstrap intervals condition on those selected predictions.",
                       "These intervals do not correct selection across 21 epochs or arms, and one seed cannot measure training variability.",
                       "The historical test is reused and is not accessed by this script."]}
    path=ROOT/(study.upper()+"_VALIDATION_ADDENDUM.json");save(result,path)
    write_markdown(result,ROOT/(study.upper()+"_VALIDATION_ADDENDUM.md"))
    return {"status":"complete","study":study,"json":str(path),
            "markdown":str(ROOT/(study.upper()+"_VALIDATION_ADDENDUM.md")),
            "r2_by_arm":{a:arms[a]["selected_raw_native_f"]["r2"] for a in cfg["arms"]}}
def write_markdown(result,path):
    study=result["study"];control=result["matched_control"];arms=result["arms"]
    rows=["# "+study.capitalize()+" validation comparison addendum","",
        "Primary metric: pooled raw native oscillator-strength R². Each arm is one validation-selected single model. Comparisons use the "+control+" control from the same round.","",
        "| Arm | Epoch 0 R² | Selected epoch | Selected R² | Final epoch 20 R² | ΔR² vs "+control+" | Molecule 95% CI | Connectivity 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for arm,v in arms.items():
        e=v["epoch_metrics"];m=v["bootstrap"]["molecule"]["delta_r2_2p5_50_97p5"]
        g=v["bootstrap"]["connectivity_group"]["delta_r2_2p5_50_97p5"]
        rows.append(f"| {arm} | {e['epoch0']['raw_native_f_r2']:.6f} | {e['selected_best']['epoch']} | {e['selected_best']['raw_native_f_r2']:.6f} | {e['final20']['raw_native_f_r2']:.6f} | {v['delta_r2_vs_matched_control']:+.6f} | [{m[0]:+.5f}, {m[2]:+.5f}] | [{g[0]:+.5f}, {g[2]:+.5f}] |")
    rows += ["","## Epoch checkpoints","",
        "| Arm | Stage | Epoch | Raw-f R² | Raw-f MAE | Raw-f RMSE | E MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm,v in arms.items():
        for stage in ("epoch0","selected_best","final20"):
            e=v["epoch_metrics"][stage]
            rows.append(f"| {arm} | {stage} | {e['epoch']} | {e['raw_native_f_r2']:.6f} | {e['raw_native_f_mae']:.6f} | {e['raw_native_f_rmse']:.6f} | {e['energy_mae']:.6f} |")
    ratio=result["objective_scale"]["architecture_f_weight_relative_to_pilot_direct_f_matched"]
    rows += ["",f"Architecture raw-f error has {ratio:.6f} times the weight of pilot direct_f_matched relative to the same energy term. Pilot control and weighted arms instead optimize trace losses. Compare architecture variants with architecture/original and objective-pilot variants with pilot/control. Architecture/original versus pilot/direct_f_matched is a loss-scale comparison, not an architecture gain.","",
        "The JSON includes raw-f MAE/RMSE, energy MAE at epoch 0/selected/final 20, every validation R² in the 21-epoch curve, per-state SSE contributions to pooled ΔR², fixed q90/q99 bright and complementary SSE, and A-derived-truth sensitivity.","",
        f"Validation molecules: {result['connectivity_layout']['molecules']}; connectivity groups: {result['connectivity_layout']['groups']}; repeated groups: {result['connectivity_layout']['groups_with_multiple_molecules']}; largest group: {result['connectivity_layout']['largest_group_molecules']}. Both paired bootstraps use {REPS} replicates and keep all ten states together.","",
        "Bootstrap intervals condition on the selected checkpoints. They do not adjust for selection among 21 epochs or arms, and one seed does not measure training variability. The historical test is not read."]
    path.write_text("\n".join(rows)+"\n")
def write_common(results):
    assert [r["study"] for r in results]==["pilot","architecture"]
    outputs={r["study"]:json.loads(Path(r["json"]).read_text()) for r in results}
    rows=["# Common validation comparison","","Each row is one selected single-model checkpoint. The matched control for pilot arms is pilot/control; for architecture arms it is architecture/original. Absolute scores across rounds are descriptive because the objective scale differs.","",
        "| Round | Arm | Best epoch | Raw-f R² | ΔR² vs matched control | f-error weighting vs pilot direct_f_matched |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    machine={"status":"complete","classification":"validation-only, selected single models",
        "studies":{},"limitations":["Cross-round pilot direct_f_matched versus architecture original compares loss scales, not architecture.",
            "Bootstrap intervals condition on validation-selected checkpoints and do not cover seed variability."]}
    for study in ("pilot","architecture"):
        report=outputs[study]
        machine["studies"][study]={"report_sha256":sha(next(Path(r["json"]) for r in results if r["study"]==study)),
            "matched_control":report["matched_control"],"arms":{}}
        for arm,v in report["arms"].items():
            e=v["epoch_metrics"]["selected_best"]
            weight="4.508882×" if study=="architecture" else ("1×" if arm=="direct_f_matched" else "trace loss")
            rows.append(f"| {study} | {arm} | {e['epoch']} | {e['raw_native_f_r2']:.6f} | {v['delta_r2_vs_matched_control']:+.6f} | {weight} |")
            machine["studies"][study]["arms"][arm]={"selected_epoch":e["epoch"],
                "selected_r2":e["raw_native_f_r2"],
                "delta_r2_vs_matched_control":v["delta_r2_vs_matched_control"],
                "f_error_weight_relative_to_pilot_direct_f_matched":weight}
    rows += ["","## Epoch 0, selected checkpoint and final epoch 20","",
        "| Round | Arm | Stage | Epoch | Raw-f R² | Raw-f MAE | Raw-f RMSE | E MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for study in ("pilot","architecture"):
        for arm,v in outputs[study]["arms"].items():
            for stage in ("epoch0","selected_best","final20"):
                e=v["epoch_metrics"][stage]
                rows.append(f"| {study} | {arm} | {stage} | {e['epoch']} | {e['raw_native_f_r2']:.6f} | {e['raw_native_f_mae']:.6f} | {e['raw_native_f_rmse']:.6f} | {e['energy_mae']:.6f} |")
    rows += ["","Architecture uses raw-f MSE divided by train-only variance 0.002510981244, which gives f error 4.508882 times the weight of pilot direct_f_matched (denominator 0.011321718690) relative to the same energy term. Pilot control and weighted arms use trace losses. Compare improvements within each round.","",
        "The historical equal-three ensemble is a separate benchmark and is not a matched single-model control. Bootstrap intervals in the per-round reports are conditional on checkpoint and arm selection; one seed does not measure training variability. No historical test predictions were opened."]
    save(machine,ROOT/"COMMON_COMPARISON.json")
    (ROOT/"COMMON_COMPARISON.md").write_text("\n".join(rows)+"\n")
    return {"json":str(ROOT/"COMMON_COMPARISON.json"),"markdown":str(ROOT/"COMMON_COMPARISON.md")}
def fixture(data):
    base=SRC/"runs"
    preds={}
    for name in ("mto_eta0","mto_eta01","mto_eta1"):
        p=base/name/"val_predictions.npz"
        preds[name],_=load_arrays(p,data,kind="f")
    ensemble=sum(preds.values())/3
    published=json.loads((RESEARCH/"ensemble_diagnostic/ensemble_metrics.json").read_text())
    r2=metric(data["y"],ensemble)["r2"]
    assert abs(r2-published["fixed_uniform_pair_and_all_means_validation_metrics"]["all_three_equal"]["r2"])<1e-12
    variants={"equal_three":ensemble,"identity_eta0":preds["mto_eta0"]}
    bootstrap,layout=bootstraps(data["y"],preds["mto_eta0"],variants,data["groups"])
    delta=(np.square(preds["mto_eta0"]-data["y"]).sum()-
           np.square(ensemble-data["y"]).sum())/np.square(data["y"]-data["y"].mean()).sum()
    assert abs(delta-sum(v["delta_r2_contribution_pooled_sst"]
                for v in decomposition(data,preds["mto_eta0"],ensemble)))<1e-12
    result={"status":"fixture_verified","classification":"Existing frozen validation predictions only; no new model selection",
        "source_val_prediction_hashes":{n:sha(base/n/"val_predictions.npz") for n in preds},
        "raw_label_data_hashes":data["hashes"],
        "eta0":metric(data["y"],preds["mto_eta0"]),
        "equal_three":metric(data["y"],ensemble),
        "delta_r2":float(delta),"connectivity_layout":layout,
        "bootstrap":{k:v["equal_three"] for k,v in bootstrap.items()},
        "per_state_contribution_sum_matches_delta_r2":True,
        "common_thresholds":{"q90":float(np.quantile(data["y"],.9)),
            "q99":float(np.quantile(data["y"],.99))}}
    path=ROOT/"FIXTURE_VALIDATION.json";save(result,path)
    return {"status":"fixture_verified","path":str(path),"delta_r2":float(delta),
            "connectivity_groups":layout["groups"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--study",choices=("fixtures","pilot","architecture","all"),required=True)
    args=ap.parse_args()
    if args.study=="fixtures":
        print(json.dumps(fixture(frozen_data()),allow_nan=False));return
    studies=("pilot","architecture") if args.study=="all" else (args.study,)
    pendings=[p for p in (ready(study) for study in studies) if p]
    if pendings:
        print(json.dumps({"status":"pending","studies":pendings,
            "note":"No running history, prediction, or checkpoint was opened."}))
        return
    data=frozen_data()
    results=[one_study(study,data) for study in studies]
    common=write_common(results) if args.study=="all" else None
    print(json.dumps({"status":"complete","results":results,"common":common},allow_nan=False))
if __name__=="__main__":main()

