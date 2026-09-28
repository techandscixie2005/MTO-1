#!/usr/bin/env python3
"""Frozen Round09 geometry on completed saved validation predictions only."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928")
HERE=ROOT/"postrun_geometry"
ROUND=ROOT/"geometry_error_audit"
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
IDENT=Path("/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json")
SEED=ROOT/"eta0_seed_replication"
SCRATCH=ROOT/"scratch_readout_comparison"
sys.path.insert(0,str(ROUND))
from geometry_error_audit import geometry,npz_member_memmap,stream_selected_identity_rows,sha
ROUND_CODE_SHA="cda3735529a57e83a6183ef82a85fc022022517404681bff0635a9bf91e54472"
ROUND_METRIC_SHA="b088b23a5098149c5a127623a5a9315f6d4c979a501eeb3fd0af46285eaccbbd"
SCRATCH_MEAN_AMENDMENT_SHA="dbff88c47c0b3e487e30507ba2f1903364f7134082abb2a3b1059bba7b4566df"
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)
def require(cond,message):
    if not cond:raise RuntimeError(message)
def pinned_round():
    require(sha(ROUND/"geometry_error_audit.py")==ROUND_CODE_SHA,"Round09 code hash changed")
    require(sha(ROUND/"geometry_error_metrics.json")==ROUND_METRIC_SHA,"Round09 metrics hash changed")
    old=json.loads((ROUND/"geometry_error_metrics.json").read_text(encoding="utf-8"))
    require(old["classification"].startswith("validation geometry"),"Round09 classification changed")
    pins=old["pinned_sources"]
    for path,digest in pins.items():
        require(sha(path)==digest,f"Round09 input hash changed: {path}")
    return old,pins
def validation_rows(old):
    ds=SRC/"data/dataset.npz";raw=SRC/"data/raw_labels.npz"
    val=np.asarray(npz_member_memmap(ds,"val"),dtype=np.int64)
    ids=np.asarray(npz_member_memmap(ds,"ids")[val])
    rawids=np.asarray(npz_member_memmap(raw,"ids")[val])
    y=np.asarray(npz_member_memmap(raw,"f")[val],dtype=np.float64)
    masks=np.asarray(npz_member_memmap(raw,"mask_f")[val],dtype=bool)
    require(len(val)==6686 and y.shape==(6686,10) and masks.all(),"validation population/masks changed")
    require(np.array_equal(ids,rawids) and np.unique(val).size==len(val),"raw IDs/indices changed")
    require(np.isfinite(y).all(),"nonfinite raw truth")
    zz=np.asarray(npz_member_memmap(ds,"z")[val])
    pp=np.asarray(npz_member_memmap(ds,"pos")[val])
    geom=[geometry(z,p) for z,p in zip(zz,pp)]
    q2=np.asarray([g[0] for g in geom]);q3=np.asarray([g[1] for g in geom])
    oldgeo=old["geometry"]
    b2=np.asarray(oldgeo["deduplicated_boundaries_including_fixed_threshold"],dtype=np.float64)
    require(np.all(np.diff(b2)>0),"nonmonotone frozen q2 cuts")
    bin2=np.where(np.isfinite(q2),np.searchsorted(b2,q2,side="left"),-1)
    require(np.count_nonzero(bin2<0)==oldgeo["validation_degenerate_count"],"q2 degenerate count differs")
    counts=[int(np.count_nonzero(bin2==j)) for j in range(len(b2)+1)]
    require(counts==[r["validation_molecules"] for r in old["strata"]],"Round09 q2 bin counts differ")
    # Identity parser lexically skips every non-validation row, including test.
    ident,nrows=stream_selected_identity_rows(IDENT,val)
    require(all(int(ident[int(i)][0])==int(mid) for i,mid in zip(val,ids)),"identity/ID mismatch")
    case=np.flatnonzero(ids==14562)
    require(len(case)==1,"case14562 absent/duplicated")
    case=int(case[0])
    require(int(val[case])==old["case_14562"]["dataset_index"],"case index differs")
    require(abs(q2[case]-old["case_14562"]["q2_lambda2_over_lambda1"])<1e-12,"case q2 differs")
    six=np.flatnonzero((bin2==0)|(bin2==1))
    require(len(six)==6 and case in six,"fixed six-case group differs")
    return {"indices":val,"ids":ids,"truth":y,"mask":masks,"q2":q2,"q3":q3,
        "bin2":bin2,"bounds2":b2,
        "case":case,"six":six,"identity_val_rows_verified":len(ident),
        "identity_nonval_rows_lexically_skipped":nrows-len(ident)}
def load_pred(path,rows,expected_sha):
    require(path.is_file(),f"PENDING: missing saved validation array {path}")
    require(sha(path)==expected_sha,f"saved array hash mismatch: {path}")
    with np.load(path,allow_pickle=False) as z:
        require(all(k in z.files for k in ("indices","ids","f_true","f")),"saved array keys missing")
        ix=z["indices"];ids=z["ids"];truth=z["f_true"];pred=z["f"].astype(np.float64)
        masks=z["mask_f_true"] if "mask_f_true" in z.files else None
    require(np.array_equal(ix,rows["indices"]) and np.array_equal(ids,rows["ids"]),"validation IDs/indices mismatch")
    require(truth.dtype==np.float64 and np.array_equal(truth,rows["truth"]),"raw FP64 truth mismatch")
    if masks is not None:require(np.array_equal(masks,rows["mask"]),"f masks mismatch")
    require(pred.shape==truth.shape and np.isfinite(pred).all(),"prediction shape/finiteness mismatch")
    return pred
def completed_inputs(rows,family):
    require(family in ("seed","scratch","all"),"unknown completed family")
    wants_seed=family in ("seed","all");wants_scratch=family in ("scratch","all")
    # All required family terminal receipts must exist before reading that family's predictions.
    fits={}
    if wants_seed:
        for seed in (23,37):
            base=SEED/f"runs/seed{seed}"
            require(not any((base/n).exists() for n in ("FAILED.json","INVALID.json")),f"seed{seed} failed/invalid marker exists")
            p=base/"FIT_COMPLETE.json"
            require(p.is_file(),f"PENDING: seed{seed} FIT_COMPLETE absent")
            d=json.loads(p.read_text(encoding="utf-8"))
            require(d.get("event")=="FIT_COMPLETE" and d.get("seed")==seed and
                    d.get("epochs")==100 and d.get("steps")==188100 and
                    d.get("best_legacy_epoch") in range(1,101) and
                    d.get("best_raw_f_epoch") in range(1,101),
                    f"PENDING: seed{seed} terminal receipt incomplete")
            fits[f"seed{seed}"]={"receipt":d,"sha256":sha(p)}
    if wants_scratch:
        for arm in ("native","mto"):
            base=SCRATCH/f"runs/{arm}"
            require(not any((base/n).exists() for n in ("FAILED.json","INVALID.json")),f"scratch {arm} failed/invalid marker exists")
            p=base/"FIT_COMPLETE.json"
            require(p.is_file(),f"PENDING: scratch {arm} FIT_COMPLETE absent")
            d=json.loads(p.read_text(encoding="utf-8"))
            require(d.get("event")=="FIT_COMPLETE" and d.get("arm")==arm and
                    d.get("epochs")==100 and d.get("steps")==188100 and
                    d.get("best",{}).get("raw_f",{}).get("epoch") in range(101),
                    f"PENDING: scratch {arm} terminal receipt incomplete")
            fits[f"scratch_{arm}"]={"receipt":d,"sha256":sha(p)}
    # Bind receipt selection to reviewed source map and immutable small fixed metrics.
    requested=[]
    if wants_seed:requested.append((SEED,("seed23","seed37")))
    if wants_scratch:requested.append((SCRATCH,("scratch_native","scratch_mto")))
    for project,names in requested:
        review=json.loads((project/"IMPLEMENTATION_REVIEW.json").read_text(encoding="utf-8"))
        require(review.get("passed") is True and review["preflight_sha256"]==sha(project/"PREFLIGHT.json"),
                f"{project.name} independent review/preflight mismatch")
        for name in names:
            fit=fits[name]["receipt"]
            require(fit["source_hashes"]==review["source_hashes"] and
                    fit["code_hashes"]==review["code_hashes"],f"{name} reviewed source map mismatch")
            require(fit.get("test_batches")==0,f"{name} test-use claim mismatch")
            base=(SEED/f"runs/{name}") if name.startswith("seed") else (SCRATCH/f"runs/{name.removeprefix('scratch_')}")
            fixed_path=base/"FIXED_CHECKPOINT_METRICS.json"
            require(fixed_path.is_file(),f"PENDING: missing {fixed_path}")
            require(sha(fixed_path)==fit["fixed_metrics_sha256"],f"{name} fixed metrics hash mismatch")
            fixed=json.loads(fixed_path.read_text(encoding="utf-8"))
            fits[name]["fixed"]=fixed
            if name.startswith("seed"):
                require(fixed["legacy"]["epoch"]==fit["best_legacy_epoch"] and
                    fixed["raw_f"]["epoch"]==fit["best_raw_f_epoch"],f"{name} selected epochs mismatch")
            else:
                require(fixed["raw_f"]["epoch"]==fit["best"]["raw_f"]["epoch"] and
                    fixed["joint"]["epoch"]==fit["best"]["joint"]["epoch"],f"{name} selected epochs mismatch")
    preds={};file_hashes={}
    if wants_seed:
        for seed in (23,37):
            d=fits[f"seed{seed}"]["receipt"];base=SEED/f"runs/seed{seed}"
            for label,name in (("legacy_joint_selected","best_legacy"),
                               ("secondary_raw_f_selected","best_raw_f")):
                path=base/f"val_{name}.npz"
                require(path.is_file(),f"PENDING: missing seed{seed} {name} predictions")
                require(sha(path)==d["val_prediction_hashes"][name],
                        f"seed{seed} selection array hash mismatch")
                key=f"seed{seed}_{label}"
                preds[key]=load_pred(path,rows,d["val_prediction_hashes"][name])
                observed=float(np.square(preds[key]-rows["truth"]).sum())
                if name=="best_raw_f":
                    require(abs(observed-d["best_raw_f_sse"])<1e-8,f"seed{seed} raw-f-selected SSE mismatch")
                else:
                    recorded=fits[f"seed{seed}"]["fixed"]["legacy"]["val"]["raw_native_f"]["sse"]
                    require(abs(observed-recorded)<1e-3,f"seed{seed} legacy-selected SSE replay mismatch")
                file_hashes[key]=sha(path)
    if wants_scratch:
        for arm in ("native","mto"):
            d=fits[f"scratch_{arm}"]["receipt"];path=SCRATCH/f"runs/{arm}/val_best_raw_f.npz"
            require(d["best"]["raw_f"]["epoch"] in range(101),"scratch selection epoch invalid")
            require(path.is_file(),f"PENDING: scratch {arm} saved predictions absent")
            require(sha(path)==d["prediction_hashes"]["best_raw_f"],
                    f"scratch {arm} selection array hash mismatch")
            key=f"scratch_{arm}_raw_f_selected"
            preds[key]=load_pred(path,rows,d["prediction_hashes"]["best_raw_f"])
            observed=float(np.square(preds[key]-rows["truth"]).sum())
            require(abs(observed-d["best"]["raw_f"]["metric"])<1e-8,
                    f"scratch {arm} selected raw-f SSE mismatch")
            file_hashes[key]=sha(path)
    return preds,{k:v["sha256"] for k,v in fits.items()},file_hashes
def metric(y,p):
    err=p-y;sse=float(np.square(err).sum());sst=float(np.square(y-y.mean()).sum())
    return {"molecules":len(y),"labels":y.size,"sse":sse,"sst":sst,
        "r2":float(1-sse/sst),"mae":float(np.abs(err).mean()),
        "rmse":float(np.sqrt(sse/y.size)),"mean_signed_error":float(err.mean()),
        "overprediction_fraction":float((err>0).mean())}
def state_rows(y,p,ref_eta=None,ref_eq=None):
    out=[]
    for j in range(10):
        err=p[:,j]-y[:,j];sse=float(np.square(err).sum())
        row={"physical_state":j+1,"labels":len(y),"sse":sse,
            "mean_signed_error":float(err.mean()),"overprediction_fraction":float((err>0).mean())}
        if ref_eta is not None:
            se=float(np.square(ref_eta[:,j]-y[:,j]).sum())
            sq=float(np.square(ref_eq[:,j]-y[:,j]).sum())
            row["delta_sse_vs_eta0"]=sse-se;row["abs_delta_sse_vs_eta0"]=abs(sse-se)
            row["delta_sse_vs_equal3"]=sse-sq;row["abs_delta_sse_vs_equal3"]=abs(sse-sq)
        out.append(row)
    return out
def bin_rows(rows,p,y,axis,ref_eta,ref_eq):
    bins=rows[f"bin{axis}"];bounds=rows[f"bounds{axis}"]
    result=[]
    for j in [-1,*range(len(bounds)+1)]:
        sel=bins==j
        if j==-1:label="degenerate"
        elif j==0:label=f"q{axis} <= {bounds[0]:.12g}"
        elif j==len(bounds):label=f"q{axis} > {bounds[-1]:.12g}"
        else:label=f"{bounds[j-1]:.12g} < q{axis} <= {bounds[j]:.12g}"
        if not sel.any():
            result.append({"bin_index":j,"label":label,"molecules":0,"labels":0,"sse":0.0,
                "per_state":[]});continue
        yy=y[sel];pp=p[sel];e=pp-yy
        sse=float(np.square(e).sum())
        se=float(np.square(ref_eta[sel]-yy).sum());sq=float(np.square(ref_eq[sel]-yy).sum())
        result.append({"bin_index":j,"label":label,"molecules":int(sel.sum()),"labels":int(yy.size),
            "sse":sse,"sse_share_of_full_validation":None,
            "q3_median_descriptive":float(np.median(rows["q3"][sel][np.isfinite(rows["q3"][sel])])) if np.isfinite(rows["q3"][sel]).any() else None,
            "mean_signed_error":float(e.mean()),"overprediction_fraction":float((e>0).mean()),
            "delta_sse_vs_eta0":sse-se,"abs_delta_sse_vs_eta0":abs(sse-se),
            "delta_sse_vs_equal3":sse-sq,"abs_delta_sse_vs_equal3":abs(sse-sq),
            "per_state":state_rows(yy,pp,ref_eta[sel],ref_eq[sel])})
    return result
def analyze(rows,preds,old,pins,mode,receipt_hashes=None,new_file_hashes=None):
    y=rows["truth"];eta=preds["eta0_legacy_joint_selected"];eq=preds["fixed_equal3"]
    summary={}
    for name,p in preds.items():
        m=metric(y,p)
        m["delta_r2_vs_eta0"]=m["r2"]-metric(y,eta)["r2"]
        m["delta_r2_vs_equal3"]=m["r2"]-metric(y,eq)["r2"]
        m["delta_sse_vs_eta0"]=m["sse"]-metric(y,eta)["sse"]
        m["abs_delta_sse_vs_eta0"]=abs(m["delta_sse_vs_eta0"])
        m["delta_sse_vs_equal3"]=m["sse"]-metric(y,eq)["sse"]
        m["abs_delta_sse_vs_equal3"]=abs(m["delta_sse_vs_equal3"])
        m["per_state"]=state_rows(y,p,eta,eq)
        m["q2_bins"]=bin_rows(rows,p,y,2,eta,eq)
        for kind in ("q2_bins",):
            for b in m[kind]:b["sse_share_of_full_validation"]=b["sse"]/m["sse"] if m["sse"] else None
            require(sum(b["molecules"] for b in m[kind])==len(y),"bin partition incomplete")
            require(abs(sum(b["sse"] for b in m[kind])-m["sse"])<1e-9,"bin SSE sum mismatch")
        case=rows["case"];six=rows["six"]
        mol_sse=np.square(p-y).sum(1)
        m["case14562_concentration_only"]={"id":14562,"global_index":int(rows["indices"][case]),
            "molecule_sse":float(mol_sse[case]),"share_of_full_sse":float(mol_sse[case]/m["sse"]),
            "q2":float(rows["q2"][case]),"q3":float(rows["q3"][case]),
            "truth":y[case].tolist(),"prediction":p[case].tolist()}
        m["fixed_six_lowest_q2_concentration_only"]={"molecule_ids":rows["ids"][six].astype(int).tolist(),
            "molecules":6,"sse":float(mol_sse[six].sum()),
            "share_of_full_sse":float(mol_sse[six].sum()/m["sse"])}
        summary[name]=m
    # Round09 fixture: same raw FP64 labels, exact eta0/equal3 native-f arrays and q2 counts.
    for name,oldname in (("eta0_legacy_joint_selected","eta0"),("fixed_equal3","equal3")):
        now=summary[name];before=old["full_validation_metrics_all_molecules_retained"][oldname]
        require(abs(now["sse"]-before["sse"])<1e-10,f"Round09 {name} full SSE mismatch")
        for j,row in enumerate(old["strata"]):
            current=next(b for b in now["q2_bins"] if b["bin_index"]==j)
            require(current["molecules"]==row["validation_molecules"],f"Round09 {name} bin count mismatch")
            require(abs(current["sse"]-row["models"][oldname]["sse"])<1e-10,
                    f"Round09 {name} q2 bin SSE mismatch {j}")
    scratch_contrast=None
    if "scratch_native_raw_f_selected" in summary and "scratch_mto_raw_f_selected" in summary:
        n=summary["scratch_native_raw_f_selected"];m=summary["scratch_mto_raw_f_selected"]
        scratch_contrast={"primary_native_minus_mto_delta_r2":n["r2"]-m["r2"],
            "primary_native_minus_mto_delta_sse":n["sse"]-m["sse"],
            "primary_native_minus_mto_abs_delta_sse":abs(n["sse"]-m["sse"])}
        if "scratch_fixed_equal_mean_secondary" in summary:
            z=summary["scratch_fixed_equal_mean_secondary"]
            scratch_contrast["postlaunch_preoutcome_secondary_equal_mean"]={
                name:{"delta_r2":z["r2"]-summary[name]["r2"],
                    "delta_sse":z["sse"]-summary[name]["sse"],
                    "abs_delta_sse":abs(z["sse"]-summary[name]["sse"])}
                for name in ("scratch_native_raw_f_selected","scratch_mto_raw_f_selected",
                    "eta0_legacy_joint_selected","fixed_equal3")}
    result={"classification":"Validation-only saved-prediction geometry comparison; no inference/fitting/test/filtering",
        "mode":mode,"round09_metrics_sha256":ROUND_METRIC_SHA,"round09_code_sha256":ROUND_CODE_SHA,
        "protocol_sha256":sha(HERE/"PROTOCOL.md"),"code_sha256":sha(__file__),
        "scratch_mean_secondary_amendment_sha256":SCRATCH_MEAN_AMENDMENT_SHA if "scratch_fixed_equal_mean_secondary" in preds else None,
        "round09_source_hashes":pins,"completed_receipt_hashes":receipt_hashes or {},
        "new_prediction_hashes":new_file_hashes or {},
        "validation_molecules":len(y),"validation_labels":int(y.size),
        "q2_train_derived_boundaries":rows["bounds2"].tolist(),
        "q3_round09_training_quantiles_diagnostic_only":old["geometry"]["training_q3_quantiles"],
        "q3_validation_quantiles_diagnostic_only":old["geometry"]["validation_q3_quantiles"],
        "interval_convention":"right-closed; undefined/degenerate separate",
        "identity_val_rows_verified":rows["identity_val_rows_verified"],
        "identity_nonval_rows_lexically_skipped":rows["identity_nonval_rows_lexically_skipped"],
        "predetermined_prediction_selection_rules":{
            "eta0_legacy_joint_selected":"historical eta0 legacy joint-objective checkpoint",
            "fixed_equal3":"equal mean historical eta0/eta01/eta1 native-f arrays",
            "seed23_legacy_joint_selected":"historical-style legacy joint objective; only in completed mode",
            "seed37_legacy_joint_selected":"historical-style legacy joint objective; only in completed mode",
            "primary_equal_seed11_23_37":"mean of seed11/23/37 LEGACY-JOINT selected native-f arrays",
            "seed23_secondary_raw_f_selected":"validation raw-f selected secondary, not primary",
            "seed37_secondary_raw_f_selected":"validation raw-f selected secondary, not primary",
            "secondary_mixed_selection_equal3":"seed11 legacy plus seed23/37 raw-f selected; mixed selection",
            "scratch_native_raw_f_selected":"raw-f SSE selected among epochs0..100",
            "scratch_mto_raw_f_selected":"raw-f SSE selected among epochs0..100",
            "scratch_fixed_equal_mean_secondary":"post-launch/pre-outcome fixed 0.5 native + 0.5 MTO selected direct-f arrays; secondary"},
        "predictors":summary,"scratch_primary_and_secondary_contrast":scratch_contrast,
        "fixture_round09_full_and_q2_bin_reproduction_passed":True,
        "limitations":["All full benchmarks retain every molecule; case14562/six-case values are concentration diagnostics only.",
            "Geometry associations are descriptive and do not establish causality.",
            "New checkpoints were selected on the same reused validation set; deltas are exploratory and selection conditional.",
            "Round09 defined q2 bins only; q3 remains a ratio/quantile diagnostic without new bins.",
            "No test indices, labels, geometry, predictions or metrics were accessed."],
        "time":time.time()}
    return result
def report(result,path):
    r=result["predictors"];lines=["# Post-run validation geometry comparison","",
        "Mode: "+result["mode"]+". All benchmark metrics retain the full validation set (6,686 molecules, 66,860 raw-f labels).",
        "No fitting, model inference, checkpoint selection, sample removal or test access occurred.","",
        "## Full validation metrics","",
        "| Predictor and selection | SSE | R² | ΔR² vs eta0 | ΔR² vs fixed equal3 | MAE | RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in r.items():
        lines.append(f"| {name} | {m['sse']:.9f} | {m['r2']:.6f} | {m['delta_r2_vs_eta0']:+.6f} | {m['delta_r2_vs_equal3']:+.6f} | {m['mae']:.7f} | {m['rmse']:.7f} |")
    if result["scratch_primary_and_secondary_contrast"] is not None:
        c=result["scratch_primary_and_secondary_contrast"]
        lines.extend(["","## Matched scratch contrast","",
            f"Primary native minus MTO selected raw-f R²: {c['primary_native_minus_mto_delta_r2']:+.6f}; ΔSSE {c['primary_native_minus_mto_delta_sse']:+.9f}."])
        if "postlaunch_preoutcome_secondary_equal_mean" in c:
            lines.append("The fixed 0.5 native + 0.5 MTO mean is a post-launch, pre-outcome secondary diagnostic; component and reference deltas are in JSON.")
    lines.extend(["","## Frozen geometry bins","",
        f"q2 cuts: {result['q2_train_derived_boundaries']}. These are from TRAIN only and right-closed; q3 has no Round09 bins.",
        f"Round09 TRAIN q3 quantiles (descriptive only): {result['q3_round09_training_quantiles_diagnostic_only']}.",
        "Round09 eta0/equal3 q2 molecule counts and SSE reproduce exactly within 1e-10.",""])
    for axis in (2,):
        lines.extend([f"### q{axis} bins","",
            "| Predictor | Bin | Molecules | SSE | ΔSSE vs eta0 | ΔSSE vs equal3 |",
            "|---|---|---:|---:|---:|---:|"])
        for name,m in r.items():
            for b in m[f"q{axis}_bins"]:
                lines.append(f"| {name} | {b['label']} | {b['molecules']} | {b['sse']:.8f} | {b.get('delta_sse_vs_eta0',0):+.8f} | {b.get('delta_sse_vs_equal3',0):+.8f} |")
        lines.append("")
    lines.extend(["## Interpretation limits","",*["- "+x for x in result["limitations"]],"",
        "Per-state SSE, signed errors, absolute delta SSE, case14562 and fixed six-case concentration are in the JSON."])
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
def main():
    ap=argparse.ArgumentParser()
    x=ap.add_mutually_exclusive_group(required=True)
    x.add_argument("--fixture",action="store_true")
    x.add_argument("--seed-completed",action="store_true")
    x.add_argument("--scratch-completed",action="store_true")
    x.add_argument("--completed",action="store_true")
    a=ap.parse_args()
    old,pins=pinned_round();rows=validation_rows(old)
    p0=SRC/"runs/mto_eta0/val_predictions.npz"
    p01=SRC/"runs/mto_eta01/val_predictions.npz"
    p1=SRC/"runs/mto_eta1/val_predictions.npz"
    preds0=[load_pred(p,rows,pins[str(p)]) for p in (p0,p01,p1)]
    preds={"eta0_legacy_joint_selected":preds0[0],"fixed_equal3":sum(preds0)/3}
    receipts=None;newfiles=None;mode="fixture eta0/equal3 only"
    family=("all" if a.completed else ("seed" if a.seed_completed else ("scratch" if a.scratch_completed else None)))
    if family:
        reviewpath=HERE/"IMPLEMENTATION_REVIEW.json"
        require(reviewpath.is_file(),"PENDING: postprocessor independent review absent")
        review=json.loads(reviewpath.read_text(encoding="utf-8"))
        require(review.get("passed") is True and review.get("code_sha256")==sha(__file__) and
            review.get("protocol_sha256")==sha(HERE/"PROTOCOL.md") and
            review.get("fixture_sha256")==sha(HERE/"FIXTURE_VALIDATION.json") and
            (family=="seed" or review.get("scratch_amendment_sha256")==SCRATCH_MEAN_AMENDMENT_SHA),
            "postprocessor exact-hash independent review mismatch")
        new,receipts,newfiles=completed_inputs(rows,family)
        preds.update(new)
        if family in ("seed","all"):
            preds["primary_equal_seed11_23_37"]=(preds["eta0_legacy_joint_selected"]+
                preds["seed23_legacy_joint_selected"]+preds["seed37_legacy_joint_selected"])/3
            preds["secondary_mixed_selection_equal3"]=(preds["eta0_legacy_joint_selected"]+
                preds["seed23_secondary_raw_f_selected"]+preds["seed37_secondary_raw_f_selected"])/3
        if family in ("scratch","all"):
            require(sha(HERE/"SCRATCH_MEAN_SECONDARY_AMENDMENT.md")==SCRATCH_MEAN_AMENDMENT_SHA,
                    "scratch mean secondary amendment changed")
            preds["scratch_fixed_equal_mean_secondary"]=(preds["scratch_native_raw_f_selected"]+
                preds["scratch_mto_raw_f_selected"])/2
        mode={"seed":"completed seed family","scratch":"completed scratch family","all":"completed seed and scratch studies"}[family]
    result=analyze(rows,preds,old,pins,mode,receipts,newfiles)
    if a.fixture:
        jsonpath=HERE/"FIXTURE_VALIDATION.json";mdpath=HERE/"FIXTURE_VALIDATION.md"
    elif a.seed_completed:
        jsonpath=HERE/"SEED_RESULTS.json";mdpath=HERE/"SEED_REPORT.md"
    elif a.scratch_completed:
        jsonpath=HERE/"SCRATCH_RESULTS.json";mdpath=HERE/"SCRATCH_REPORT.md"
    else:
        jsonpath=HERE/"COMPLETED_RESULTS.json";mdpath=HERE/"COMPLETED_REPORT.md"
    if family:require(not jsonpath.exists() and not mdpath.exists(),"completed output already exists; preserve prior result")
    atomic_json(jsonpath,result);report(result,mdpath)
    check=json.loads(jsonpath.read_text(encoding="utf-8"))
    require(check["fixture_round09_full_and_q2_bin_reproduction_passed"],"readback fixture failed")
    require("Full validation metrics" in mdpath.read_text(encoding="utf-8"),"Markdown readback failed")
    print(json.dumps({"mode":mode,"predictors":len(preds),"full_eta0_sse":result["predictors"]["eta0_legacy_joint_selected"]["sse"],
        "full_equal3_sse":result["predictors"]["fixed_equal3"]["sse"],
        "report":str(mdpath),"results":str(jsonpath)}))
if __name__=="__main__":
    try:main()
    except RuntimeError as e:
        print(str(e),file=sys.stderr)
        sys.exit(2)

