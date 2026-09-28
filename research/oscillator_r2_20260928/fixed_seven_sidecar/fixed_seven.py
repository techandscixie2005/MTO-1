#!/usr/bin/env python3
"""Reviewed saved-array fixed-seven secondary and scratch F5 reference; no inference."""
import argparse, hashlib, json, os, sys, tempfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928")
HERE=ROOT/"fixed_seven_sidecar"
PGROOT=ROOT/"postrun_geometry"
sys.path.insert(0,str(PGROOT))
import postrun_geometry as pg
SEED=ROOT/"eta0_seed_replication"
SCRATCH=ROOT/"scratch_readout_comparison"
AMENDMENT_SHA="675030449886676094b1226d6ea75d753752fdd7e3e7cc3c8e1a9441bb469c0e"
PROVENANCE_SHA="6a1ef5ae13b7c1675fed75c0461c8a5d1d4c0f596e1029d90b263411cc43201b"
SCRATCH_PINS_SHA="c4db1881dc5b6c89e73bfacac1e9338ec68d3629ee45983d237973da373cf302"
EXPECTED_F5_R2=0.48769960489832564
Q90,Q99=.0546,.2377

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path,obj):
    require(not path.exists(),f"completed output exists: {path}")
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temp,path)
def static_pins():
    require(sha(PGROOT/"FIXED_SEVEN_SECONDARY_AMENDMENT.md")==AMENDMENT_SHA,"fixed7 amendment changed")
    require(sha(PGROOT/"FIXED_SEVEN_SECONDARY_PROVENANCE.json")==PROVENANCE_SHA,"fixed7 provenance changed")
    provenance=read_json(PGROOT/"FIXED_SEVEN_SECONDARY_PROVENANCE.json")
    require(provenance["protocol_sha256"]==AMENDMENT_SHA,"fixed7 protocol binding")
    for rel,digest in provenance["reference_hashes"].items():
        require(sha(ROOT/rel)==digest,f"reference changed: {rel}")
    for label,item in provenance["frozen_five_components"].items():
        require(sha(item["prediction_path"])==item["prediction_sha256"],f"frozen component changed: {label}")
    old,pins=pg.pinned_round()
    rows=pg.validation_rows(old)
    return provenance,old,pins,rows

def load_strict(path,rows,expected):
    require(path.is_file(),f"PENDING: missing saved validation prediction {path}")
    require(sha(path)==expected,f"prediction SHA changed: {path}")
    with np.load(path,allow_pickle=False) as z:
        require(set(("indices","ids","f_true","mask_f_true","f")).issubset(z.files),
                f"saved prediction keys incomplete: {path}")
        ix=z["indices"];ids=z["ids"];truth=z["f_true"];masks=z["mask_f_true"];pred=z["f"]
    require(np.array_equal(ix,rows["indices"]) and np.array_equal(ids,rows["ids"]),
            f"saved validation index/ID mismatch: {path}")
    require(truth.dtype==np.float64 and np.array_equal(truth,rows["truth"]),
            f"raw FP64 printed truth mismatch: {path}")
    require(np.array_equal(masks,rows["mask"]) and masks.dtype==np.bool_,
            f"raw label masks mismatch: {path}")
    require(pred.shape==truth.shape and np.issubdtype(pred.dtype,np.floating) and
            np.all(np.isfinite(pred)),"prediction shape/dtype/finiteness")
    return np.asarray(pred,dtype=np.float64)

def load_five(provenance,rows):
    five={}
    for label,item in provenance["frozen_five_components"].items():
        five[label]=load_strict(Path(item["prediction_path"]),rows,item["prediction_sha256"])
    require(tuple(five)==("mto_eta0","mto_eta01","mto_eta1","G1","G3"),"five component order")
    F5=sum(five.values())/5.0
    require(abs(metric(rows["truth"],F5)["r2"]-EXPECTED_F5_R2)<1e-11,"frozen F5 R2 mismatch")
    return five,F5

def metric(y,p):
    require(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),"invalid metric input")
    e=p-y;sse=float(np.square(e).sum(dtype=np.float64))
    sst=float(np.square(y-y.mean()).sum(dtype=np.float64))
    require(sst>0,"zero SST")
    return {"molecules":len(y),"transitions":int(y.size),"sse":sse,"sst":sst,
            "r2":float(1-sse/sst),"rmse":float(np.sqrt(sse/y.size)),
            "mae":float(np.abs(e).mean())}

def check_terminal(d,seed):
    require(d.get("event")=="FIT_COMPLETE" and d.get("seed")==seed,
            f"seed{seed} terminal event/identity")
    require(d.get("epochs")==100 and d.get("steps")==188100 and d.get("test_batches")==0,
            f"seed{seed} incomplete or test-used")
    require(type(d.get("best_legacy_epoch")) is int and 1<=d["best_legacy_epoch"]<=100,
            f"seed{seed} legacy epoch invalid")
    require(type(d.get("best_raw_f_epoch")) is int and 1<=d["best_raw_f_epoch"]<=100,
            f"seed{seed} secondary epoch invalid")

def seed_terminal_pair(project):
    """Read both terminal receipts and fail before any selected prediction is opened."""
    fits={}
    for seed in (23,37):
        base=project/f"runs/seed{seed}"
        require(not any((base/x).exists() for x in ("FAILED.json","INVALID.json")),
                f"seed{seed} FAILED/INVALID marker")
        path=base/"FIT_COMPLETE.json"
        require(path.is_file(),f"PENDING: seed{seed} FIT_COMPLETE absent")
        fit=read_json(path);check_terminal(fit,seed)
        fits[seed]={"receipt":fit,"terminal_sha256":sha(path)}
    return fits

def selected_seed_artifact(base,fit,rows,seed):
    fixed_path=base/"FIXED_CHECKPOINT_METRICS.json"
    require(fixed_path.is_file() and sha(fixed_path)==fit["fixed_metrics_sha256"],
            f"seed{seed} fixed metrics binding")
    fixed=read_json(fixed_path)
    require(fixed["legacy"]["epoch"]==fit["best_legacy_epoch"] and
            abs(fixed["legacy"]["val"]["legacy_val_objective"][0]-fit["best_legacy"])<=1e-5,
            f"seed{seed} legacy selection/fixed metrics mismatch")
    checkpoint=base/"best_legacy.pt"
    require(checkpoint.is_file() and sha(checkpoint)==fit["checkpoint_hashes"]["legacy"],
            f"seed{seed} legacy checkpoint hash mismatch")
    predpath=base/"val_best_legacy.npz"
    pred=load_strict(predpath,rows,fit["val_prediction_hashes"]["best_legacy"])
    observed=metric(rows["truth"],pred)["sse"]
    require(abs(observed-fixed["legacy"]["val"]["raw_native_f"]["sse"])<=1e-3,
            f"seed{seed} fixed checkpoint raw-f SSE mismatch")
    return pred,{"fixed_metrics_sha256":fit["fixed_metrics_sha256"],
        "legacy_epoch":fit["best_legacy_epoch"],
        "legacy_checkpoint_sha256":fit["checkpoint_hashes"]["legacy"],
        "legacy_prediction_sha256":fit["val_prediction_hashes"]["best_legacy"],
        "saved_array_sse":observed}

def seed_artifacts_after_terminal(project,rows):
    fits=seed_terminal_pair(project)
    out={};evidence={}
    for seed in (23,37):
        fit=fits[seed]["receipt"]
        pred,details=selected_seed_artifact(project/f"runs/seed{seed}",fit,rows,seed)
        out[seed]=pred
        evidence[str(seed)]={"terminal_sha256":fits[seed]["terminal_sha256"],**details}
    return out,evidence,fits

def check_scratch_terminal(d,arm):
    require(d.get("event")=="FIT_COMPLETE" and d.get("arm")==arm and
            d.get("epochs")==100 and d.get("steps")==188100 and d.get("test_batches")==0,
            f"scratch {arm} terminal incomplete/identity/test-use")
    best=d.get("best",{})
    for selection in ("raw_f","joint"):
        epoch=best.get(selection,{}).get("epoch")
        require(type(epoch) is int and 0<=epoch<=100,
                f"scratch {arm} {selection} selection invalid")

def scratch_terminal_pair(project):
    """Check both scratch FIT receipts before any scratch selected array is opened."""
    fits={}
    for arm in ("native","mto"):
        base=project/f"runs/{arm}"
        require(not any((base/x).exists() for x in ("FAILED.json","INVALID.json")),
                f"scratch {arm} FAILED/INVALID marker")
        path=base/"FIT_COMPLETE.json"
        require(path.is_file(),f"PENDING: scratch {arm} FIT_COMPLETE absent")
        fit=read_json(path);check_scratch_terminal(fit,arm)
        fits[arm]={"receipt":fit,"terminal_sha256":sha(path)}
    return fits

def selected_scratch_artifact(base,fit,rows,arm):
    fixed_path=base/"FIXED_CHECKPOINT_METRICS.json"
    require(fixed_path.is_file() and sha(fixed_path)==fit["fixed_metrics_sha256"],
            f"scratch {arm} fixed metrics hash")
    fixed=read_json(fixed_path)
    epoch=fit["best"]["raw_f"]["epoch"]
    require(fixed["raw_f"]["epoch"]==epoch,"scratch raw-f selection mismatch")
    checkpoint=base/"best_raw_f.pt"
    require(checkpoint.is_file() and sha(checkpoint)==fit["checkpoint_hashes"]["raw_f"] and
            fixed["raw_f"]["checkpoint_sha256"]==fit["checkpoint_hashes"]["raw_f"],
            f"scratch {arm} selected checkpoint mismatch")
    path=base/"val_best_raw_f.npz"
    pred=load_strict(path,rows,fit["prediction_hashes"]["best_raw_f"])
    observed=metric(rows["truth"],pred)["sse"]
    require(abs(observed-fit["best"]["raw_f"]["metric"])<=1e-8,
            f"scratch {arm} selected raw-f SSE mismatch")
    return pred,{"fixed_metrics_sha256":fit["fixed_metrics_sha256"],
        "raw_f_epoch":epoch,"selected_checkpoint_sha256":fit["checkpoint_hashes"]["raw_f"],
        "selected_prediction_sha256":fit["prediction_hashes"]["best_raw_f"],
        "saved_array_sse":observed}

def scratch_artifacts_after_terminal(project,rows):
    fits=scratch_terminal_pair(project)
    out={};evidence={}
    for arm in ("native","mto"):
        fit=fits[arm]["receipt"]
        pred,details=selected_scratch_artifact(project/f"runs/{arm}",fit,rows,arm)
        out[arm]=pred
        evidence[arm]={"terminal_sha256":fits[arm]["terminal_sha256"],**details}
    return out,evidence,fits

def reviewed_execution_gate():
    p=HERE/"IMPLEMENTATION_REVIEW.json"
    require(p.is_file(),"PENDING: sidecar independent implementation review absent")
    review=read_json(p)
    require(review.get("passed") is True and review.get("script_sha256")==sha(__file__) and
            review.get("protocol_sha256")==sha(HERE/"PROTOCOL.md") and
            review.get("preflight_sha256")==sha(HERE/"PREPARATION_PREFLIGHT.json") and
            review.get("amendment_sha256")==AMENDMENT_SHA and
            review.get("provenance_sha256")==PROVENANCE_SHA and
            review.get("scratch_pins_sha256")==SCRATCH_PINS_SHA,
            "sidecar exact-hash review mismatch")
    return review

def verify_file_map(table):
    for path,digest in table.items():
        require(sha(path)==digest,f"reviewed source changed: {path}")

def load_completed_seeds(rows):
    review=read_json(SEED/"IMPLEMENTATION_REVIEW.json")
    require(review.get("passed") is True and
            review["preflight_sha256"]==sha(SEED/"PREFLIGHT.json") and
            sha(SEED/"PROTOCOL.md")==
              read_json(PGROOT/"FIXED_SEVEN_SECONDARY_PROVENANCE.json")["reference_hashes"]["eta0_seed_replication/PROTOCOL.md"],
            "seed review/preflight/protocol changed")
    pre=read_json(SEED/"PREFLIGHT.json")
    require(pre["passed"] and pre["source_hashes"]==review["source_hashes"] and
            pre["code_hashes"]==review["code_hashes"],"seed preflight map mismatch")
    verify_file_map(review["source_hashes"])
    verify_file_map(review["code_hashes"])
    fits=seed_terminal_pair(SEED)
    # The full pair and reviewed maps are validated before either prediction is loaded.
    for seed in (23,37):
        fit=fits[seed]["receipt"]
        require(fit["source_hashes"]==review["source_hashes"] and
                fit["code_hashes"]==review["code_hashes"],
                f"seed{seed} receipt source mismatch")
    out={};evidence={}
    for seed in (23,37):
        fit=fits[seed]["receipt"]
        pred,details=selected_seed_artifact(SEED/f"runs/seed{seed}",fit,rows,seed)
        out[seed]=pred
        evidence[str(seed)]={"terminal_sha256":fits[seed]["terminal_sha256"],**details}
    return out,evidence

def identity_groups(rows):
    ident,nrows=pg.stream_selected_identity_rows(pg.IDENT,rows["indices"])
    require(len(ident)==6686 and nrows>=len(ident),"selected identity count")
    groups=defaultdict(list)
    for j,(ix,mid) in enumerate(zip(rows["indices"],rows["ids"])):
        entry=ident[int(ix)]
        require(int(entry[0])==int(mid) and entry[2] is True,"validation identity mismatch")
        groups[entry[1]].append(j)
    keys=sorted(groups)
    return [groups[k] for k in keys],keys

def bootstrap(y,ref,pred,units,seed=20260929,reps=2000):
    # The same seed and sorted unit order produce identical draws for both fixed contrasts.
    n=np.array([len(g)*10 for g in units],dtype=np.float64)
    sy=np.array([y[g].sum() for g in units],dtype=np.float64)
    sy2=np.array([np.square(y[g]).sum() for g in units],dtype=np.float64)
    sr=np.array([np.square(ref[g]-y[g]).sum() for g in units],dtype=np.float64)
    sp=np.array([np.square(pred[g]-y[g]).sum() for g in units],dtype=np.float64)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(reps):
        ix=rng.integers(0,len(units),len(units))
        denom=sy2[ix].sum()-sy[ix].sum()**2/n[ix].sum()
        require(denom>0,"bootstrap zero SST")
        values.append(float((sr[ix].sum()-sp[ix].sum())/denom))
    vals=np.asarray(values)
    return {"units":len(units),"replicates":reps,"seed":seed,
            "delta_r2_ci95":np.quantile(vals,[.025,.5,.975]).tolist(),
            "positive_fraction":float((vals>0).mean())}

def slices(rows,ref,pred):
    y=rows["truth"];sst=metric(y,ref)["sst"]
    states=[]
    for j in range(10):
        a=float(np.square(ref[:,j]-y[:,j]).sum());b=float(np.square(pred[:,j]-y[:,j]).sum())
        states.append({"physical_state":j+1,"state_index_zero_based":j,"labels":len(y),
                       "reference_sse":a,"candidate_sse":b,"delta_sse_candidate_minus_reference":b-a,
                       "delta_r2_pooled_contribution":(a-b)/sst})
    tails={}
    for label,sel in (("below_q90",y<Q90),("q90_and_above",y>=Q90),("q99_and_above",y>=Q99)):
        a=float(np.square(ref[sel]-y[sel]).sum());b=float(np.square(pred[sel]-y[sel]).sum())
        tails[label]={"count":int(sel.sum()),"reference_sse":a,"candidate_sse":b,
                      "delta_sse_candidate_minus_reference":b-a}
    return states,tails

def concentration(rows,ref,pred,group_rows,group_keys):
    y=rows["truth"]
    refmol=np.square(ref-y).sum(1);predmol=np.square(pred-y).sum(1)
    gain=refmol-predmol
    net=float(gain.sum());positive=np.clip(gain,0,None);negative=np.clip(gain,None,0)
    best=np.argsort(-gain,kind="stable");worst=np.argsort(gain,kind="stable")
    def row(j):
        return {"id":int(rows["ids"][j]),"dataset_index":int(rows["indices"][j]),
                "reference_sse":float(refmol[j]),"candidate_sse":float(predmol[j]),
                "sse_gain_reference_minus_candidate":float(gain[j]),
                "share_of_net_gain":float(gain[j]/net) if net>0 else None}
    group_ref=np.asarray([refmol[g].sum() for g in group_rows])
    group_pred=np.asarray([predmol[g].sum() for g in group_rows])
    group_gain=group_ref-group_pred
    def group(j):
        ids=rows["ids"][group_rows[j]]
        return {"connectivity_key_sha256":hashlib.sha256(str(group_keys[j]).encode()).hexdigest(),
                "molecule_count":len(ids),"molecule_ids":ids.astype(int).tolist(),
                "reference_sse":float(group_ref[j]),"candidate_sse":float(group_pred[j]),
                "sse_gain_reference_minus_candidate":float(group_gain[j]),
                "share_of_net_gain":float(group_gain[j]/net) if net>0 else None}
    bins=[]
    for b in [-1,*range(len(rows["bounds2"])+1)]:
        sel=rows["bin2"]==b
        bins.append({"q2_bin":b,"molecules":int(sel.sum()),
                     "reference_sse":float(refmol[sel].sum()),
                     "candidate_sse":float(predmol[sel].sum()),
                     "sse_gain_reference_minus_candidate":float(gain[sel].sum())})
    six=rows["six"];case=rows["case"]
    top_positive=[int(j) for j in best if gain[j]>0][:10]
    return {"net_sse_gain":net,"molecules_improved":int((gain>0).sum()),
            "molecules_worsened":int((gain<0).sum()),"gross_positive_gain":float(positive.sum()),
            "gross_deterioration":float(-negative.sum()),
            "largest_positive_molecule":row(int(best[0])) if gain[best[0]]>0 else None,
            "largest_negative_molecule":row(int(worst[0])) if gain[worst[0]]<0 else None,
            "top10_positive_molecules":[row(j) for j in top_positive],
            "top10_positive_gain":float(positive[top_positive].sum()),
            "top10_share_of_net_gain":float(positive[top_positive].sum()/net) if net>0 else None,
            "largest_positive_group":group(int(np.argmax(group_gain))) if group_gain.max()>0 else None,
            "largest_negative_group":group(int(np.argmin(group_gain))) if group_gain.min()<0 else None,
            "case14562":row(case),
            "six_low_q2":{"ids":rows["ids"][six].astype(int).tolist(),
                          "reference_sse":float(refmol[six].sum()),
                          "candidate_sse":float(predmol[six].sum()),
                          "sse_gain_reference_minus_candidate":float(gain[six].sum())},
            "fixed_q2_bins":bins}

def compare(rows,reference,candidate,group_rows,group_keys):
    y=rows["truth"];a=metric(y,reference);b=metric(y,candidate)
    states,tails=slices(rows,reference,candidate)
    mol=[[i] for i in range(len(y))]
    return {"reference_metrics":a,"candidate_metrics":b,
            "delta_sse_candidate_minus_reference":b["sse"]-a["sse"],
            "delta_r2_candidate_minus_reference":b["r2"]-a["r2"],
            "molecule_bootstrap":bootstrap(y,reference,candidate,mol),
            "connectivity_bootstrap":bootstrap(y,reference,candidate,group_rows),
            "per_state":states,"tails":tails,
            "concentration":concentration(rows,reference,candidate,group_rows,group_keys)}

def report(result,path):
    lines=["# Fixed-seven saved-array validation sidecar","",
        "Mode: "+result["mode"]+". Every pooled score retains all 6,686 molecules and 66,860 raw-f labels.",
        "This is a saved-array calculation; no model inference, training, fitting or test access.","",
        "## Pooled raw-f metrics","",
        "| Predictor | SSE | R² | RMSE | MAE |","|---|---:|---:|---:|---:|"]
    for name,m in result["metrics"].items():
        lines.append(f"| {name} | {m['sse']:.9f} | {m['r2']:.9f} | {m['rmse']:.7f} | {m['mae']:.7f} |")
    for name,c in result["comparisons"].items():
        lines.extend(["",f"## {name}","",
            f"ΔR² {c['delta_r2_candidate_minus_reference']:+.9f}; ΔSSE {c['delta_sse_candidate_minus_reference']:+.9f}.",
            f"Molecule CI {c['molecule_bootstrap']['delta_r2_ci95']}; connectivity-group CI {c['connectivity_bootstrap']['delta_r2_ci95']}.",
            "Per-state, fixed-tail and concentration details are in JSON."])
    counts=result["inference_forward_counts"]
    lines.extend(["","Forward counts: "+", ".join(f"{k}={v}" for k,v in counts.items())+
        ". These are not calibrated latency ratios."])
    if result["mode"]=="seed family completed":
        lines.append("The seed3 primary and mixed-selection secondary in the original seed study remain unchanged.")
    else:
        lines.append("F5 is a supplemental scratch reference; native-minus-MTO remains the scratch primary contrast.")
    lines.extend(["F5 is a provisional validation candidate; the original fixed-three historical-test score belongs only to that predictor.",
        "All intervals are descriptive conditional on reused validation selection."])
    require(not path.exists(),f"completed report exists: {path}")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def synthetic_family_gates():
    """Small synthetic saved-array fixtures; never points at active run directories."""
    def expect_failure(fn,fragment):
        try:fn()
        except RuntimeError as exc:
            require(fragment in str(exc),f"wrong synthetic failure: {exc}")
        else:raise RuntimeError(f"synthetic gate did not reject {fragment}")
    toy={"indices":np.arange(3,dtype=np.int64),
         "ids":np.array([101,102,103],dtype=np.int64),
         "truth":np.arange(30,dtype=np.float64).reshape(3,10)/100,
         "mask":np.ones((3,10),dtype=bool)}
    passes={}
    with tempfile.TemporaryDirectory(prefix="mto_fixed7_fixture_") as temporary:
        root=Path(temporary)
        def artifact(base,pred,kind):
            base.mkdir(parents=True,exist_ok=True)
            checkpoint=base/("best_legacy.pt" if kind=="seed" else "best_raw_f.pt")
            checkpoint.write_bytes(b"synthetic selected checkpoint")
            predpath=base/("val_best_legacy.npz" if kind=="seed" else "val_best_raw_f.npz")
            np.savez_compressed(predpath,ids=toy["ids"],indices=toy["indices"],
                                f_true=toy["truth"],mask_f_true=toy["mask"],f=pred)
            sse=metric(toy["truth"],pred)["sse"]
            if kind=="seed":
                fixed={"legacy":{"epoch":1,"val":{"legacy_val_objective":[.1],
                       "raw_native_f":{"sse":sse}}}}
                selection={"best_legacy_epoch":1,"best_raw_f_epoch":2,"best_legacy":.1,
                    "checkpoint_hashes":{"legacy":sha(checkpoint)},
                    "val_prediction_hashes":{"best_legacy":sha(predpath)}}
            else:
                fixed={"raw_f":{"epoch":1,"checkpoint_sha256":sha(checkpoint)}}
                selection={"best":{"raw_f":{"epoch":1,"metric":sse},"joint":{"epoch":2}},
                    "checkpoint_hashes":{"raw_f":sha(checkpoint)},
                    "prediction_hashes":{"best_raw_f":sha(predpath)}}
            fixedpath=base/"FIXED_CHECKPOINT_METRICS.json"
            fixedpath.write_text(json.dumps(fixed),encoding="utf-8")
            return {**selection,"fixed_metrics_sha256":sha(fixedpath)},checkpoint,predpath
        seedroot=root/"seeds"
        specs={}
        for seed in (23,37):
            base=seedroot/f"runs/seed{seed}"
            selection,ck,pred=artifact(base,toy["truth"]+seed/10000,"seed")
            fit={"event":"FIT_COMPLETE","seed":seed,"epochs":100,"steps":188100,
                 "test_batches":0,**selection}
            specs[seed]={"base":base,"fit":fit,"checkpoint":ck,"prediction":pred}
        specs[23]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(specs[23]["fit"]))
        expect_failure(lambda:seed_artifacts_after_terminal(seedroot,toy),"seed37 FIT_COMPLETE")
        specs[37]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(specs[37]["fit"]))
        loaded,_,_=seed_artifacts_after_terminal(seedroot,toy)
        require(set(loaded)=={23,37},"synthetic seed pair success")
        passes["seed_pair_success"]=True
        for seed,marker in ((23,"FAILED.json"),(37,"INVALID.json")):
            path=specs[seed]["base"]/marker;path.write_text("{}")
            expect_failure(lambda:seed_artifacts_after_terminal(seedroot,toy),marker.split(".")[0])
            path.unlink()
        bad=specs[37]["fit"].copy();bad["best_legacy_epoch"]=0
        specs[37]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(bad))
        expect_failure(lambda:seed_artifacts_after_terminal(seedroot,toy),"legacy epoch")
        specs[37]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(specs[37]["fit"]))
        ck=specs[23]["checkpoint"];original=ck.read_bytes();ck.write_bytes(b"wrong checkpoint")
        expect_failure(lambda:seed_artifacts_after_terminal(seedroot,toy),"checkpoint hash")
        ck.write_bytes(original)
        pred=specs[37]["prediction"];original=pred.read_bytes();pred.write_bytes(original+b"x")
        expect_failure(lambda:seed_artifacts_after_terminal(seedroot,toy),"prediction SHA")
        pred.write_bytes(original)
        passes["seed_missing_failed_invalid_selection_checkpoint_array_rejected"]=True

        scratchroot=root/"scratch";scratch={}
        for arm in ("native","mto"):
            base=scratchroot/f"runs/{arm}"
            selection,ck,pred=artifact(base,toy["truth"]+.03,"scratch")
            fit={"event":"FIT_COMPLETE","arm":arm,"epochs":100,"steps":188100,
                 "test_batches":0,**selection}
            scratch[arm]={"base":base,"fit":fit,"checkpoint":ck,"prediction":pred}
        scratch["native"]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(scratch["native"]["fit"]))
        expect_failure(lambda:scratch_artifacts_after_terminal(scratchroot,toy),"scratch mto FIT_COMPLETE")
        scratch["mto"]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(scratch["mto"]["fit"]))
        loaded,_,_=scratch_artifacts_after_terminal(scratchroot,toy)
        require(set(loaded)=={"native","mto"},"synthetic scratch pair success")
        passes["scratch_pair_success"]=True
        for arm,marker in (("native","FAILED.json"),("mto","INVALID.json")):
            path=scratch[arm]["base"]/marker;path.write_text("{}")
            expect_failure(lambda:scratch_artifacts_after_terminal(scratchroot,toy),marker.split(".")[0])
            path.unlink()
        bad=scratch["mto"]["fit"].copy()
        bad["best"]={"raw_f":{"epoch":101,"metric":0.0},"joint":{"epoch":2}}
        scratch["mto"]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(bad))
        expect_failure(lambda:scratch_artifacts_after_terminal(scratchroot,toy),"selection invalid")
        scratch["mto"]["base"].joinpath("FIT_COMPLETE.json").write_text(json.dumps(scratch["mto"]["fit"]))
        ck=scratch["native"]["checkpoint"];original=ck.read_bytes();ck.write_bytes(b"wrong checkpoint")
        expect_failure(lambda:scratch_artifacts_after_terminal(scratchroot,toy),"checkpoint")
        ck.write_bytes(original)
        pred=scratch["mto"]["prediction"];original=pred.read_bytes();pred.write_bytes(original+b"x")
        expect_failure(lambda:scratch_artifacts_after_terminal(scratchroot,toy),"prediction SHA")
        pred.write_bytes(original)
        passes["scratch_missing_failed_invalid_selection_checkpoint_array_rejected"]=True
    return passes

def fixture():
    provenance,old,pins,rows=static_pins()
    five,F5=load_five(provenance,rows)
    eta=five["mto_eta0"];equal3=(eta+five["mto_eta01"]+five["mto_eta1"])/3.0
    geometry=pg.analyze(rows,{"eta0_legacy_joint_selected":eta,"fixed_equal3":equal3,"F5":F5},
                        old,pins,"fixed7 sidecar fixture")
    y=np.arange(30,dtype=np.float64).reshape(3,10)/100
    components=[np.ones_like(y)*k for k in (1,2,3,4,5)]
    f5=sum(components)/5
    s23=np.ones_like(y)*6;s37=np.ones_like(y)*7
    f7=(5*f5+s23+s37)/7
    direct=sum(components+[s23,s37])/7
    seed3=(components[0]+s23+s37)/3
    require(np.array_equal(f7,direct) and np.allclose(f7,4.0),"seven equal weights")
    require(np.all(f7!=(f5+seed3)/2),"block averaging incorrectly duplicates eta0")
    invalid=[]
    for bad in ({},{"event":"FIT_COMPLETE","seed":23,"epochs":100},
                {"event":"FIT_COMPLETE","seed":23,"epochs":100,"steps":188100,
                 "test_batches":1,"best_legacy_epoch":1,"best_raw_f_epoch":1}):
        try:check_terminal(bad,23)
        except RuntimeError:invalid.append(True)
        else:invalid.append(False)
    require(all(invalid),"incomplete synthetic receipt accepted")
    family_cases=synthetic_family_gates()
    toyref=y+.2;toycand=y+.1
    toy_boot=bootstrap(y,toyref,toycand,[[0],[1],[2]],reps=200)
    require(toy_boot["positive_fraction"]==1.0,"synthetic paired bootstrap sign")
    result={"classification":"fixture-only, no seed/scratch outcome files opened",
            "time":time.time(),"protocol_sha256":sha(HERE/"PROTOCOL.md"),
            "script_sha256":sha(__file__),"amendment_sha256":AMENDMENT_SHA,
            "provenance_sha256":PROVENANCE_SHA,
            "frozen_five_prediction_hashes":{k:v["prediction_sha256"] for k,v in provenance["frozen_five_components"].items()},
            "F5_metrics":metric(rows["truth"],F5),
            "round09_eta0_equal3_reproduction_passed":geometry["fixture_round09_full_and_q2_bin_reproduction_passed"],
            "round09_q2_counts":[x["molecules"] for x in geometry["predictors"]["F5"]["q2_bins"]],
            "seven_equal_weights_synthetic_passed":True,
            "synthetic_incomplete_receipts_rejected":len(invalid),
            "synthetic_bootstrap_positive_fraction":toy_boot["positive_fraction"],
            "synthetic_full_family_gates":family_cases,
            "scratch_source_pins_sha256":SCRATCH_PINS_SHA,
            "active_outcome_access":False,"test_access":False}
    (HERE/"PREPARATION_PREFLIGHT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"fixture_passed":True,"F5_r2":result["F5_metrics"]["r2"],
                      "source_sha256":sha(__file__)}))

def completed_seed():
    reviewed_execution_gate()
    provenance,old,pins,rows=static_pins()
    five,F5=load_five(provenance,rows)
    seeds,evidence=load_completed_seeds(rows)
    seed3=(five["mto_eta0"]+seeds[23]+seeds[37])/3.0
    F7=(5*F5+seeds[23]+seeds[37])/7.0
    require(np.allclose(F7,(sum(five.values())+seeds[23]+seeds[37])/7.0,rtol=0,atol=1e-15),
            "fixed-seven algebra")
    group_rows,group_keys=identity_groups(rows)
    contrasts={"F7_minus_F5":compare(rows,F5,F7,group_rows,group_keys),
               "F7_minus_seed3_primary":compare(rows,seed3,F7,group_rows,group_keys)}
    result={"classification":"exploratory fixed-seven saved-validation-array secondary; no inference/test",
            "mode":"seed family completed","time":time.time(),
            "protocol_sha256":sha(HERE/"PROTOCOL.md"),"script_sha256":sha(__file__),
            "review_sha256":sha(HERE/"IMPLEMENTATION_REVIEW.json"),
            "amendment_sha256":AMENDMENT_SHA,"provenance_sha256":PROVENANCE_SHA,
            "seed_terminal_and_selection_evidence":evidence,
            "five_component_hashes":{k:v["prediction_sha256"] for k,v in provenance["frozen_five_components"].items()},
            "weights":"exact equal 1/7 over eta0,eta01,eta1,G1,G3,seed23_legacy,seed37_legacy",
            "inference_forward_counts":{"F7":7,"F5":5,"seed3_primary":3},
            "metrics":{"F5":metric(rows["truth"],F5),"seed3_primary":metric(rows["truth"],seed3),
                       "F7":metric(rows["truth"],F7)},
            "comparisons":contrasts,"connectivity_groups":len(group_rows),
            "fixed_q2_boundaries_train_only":rows["bounds2"].tolist(),
            "fixed_tail_thresholds":{"q90":Q90,"q99":Q99},
            "limitations":["Post-individual-result exploratory candidate on reused validation.",
                           "Bootstrap intervals omit selection/adaptation uncertainty.",
                           "No test access, model inference, fitting, filtering or mixture search.",
                           "Seven model forwards are not calibrated latency relative to five or three."]}
    write_json(HERE/"SEED_FIXED7_RESULTS.json",result)
    report(result,HERE/"SEED_FIXED7_REPORT.md")
    print(json.dumps({"F7_r2":result["metrics"]["F7"]["r2"],
                      "F5_r2":result["metrics"]["F5"]["r2"]}))

def scratch_supplemental():
    reviewed_execution_gate()
    pins_path=HERE/"SCRATCH_SOURCE_PINS.json"
    require(sha(pins_path)==SCRATCH_PINS_SHA,"scratch sidecar pin file changed")
    sp=read_json(pins_path)["scratch_source_hashes"]
    for name,digest in sp.items():
        require(sha(SCRATCH/name)==digest,f"scratch {name} changed")
    review=read_json(SCRATCH/"IMPLEMENTATION_REVIEW.json")
    pre=read_json(SCRATCH/"PREFLIGHT.json")
    require(review.get("passed") is True and pre.get("passed") is True and
            review["preflight_sha256"]==sp["PREFLIGHT.json"] and
            review["protocol_sha256"]==sp["PROTOCOL.md"] and
            pre["source_hashes"]==review["source_hashes"] and
            pre["code_hashes"]==review["code_hashes"],
            "scratch reviewed source/preflight mismatch")
    verify_file_map(review["source_hashes"])
    verify_file_map(review["code_hashes"])
    require(sha(PGROOT/"SCRATCH_MEAN_SECONDARY_AMENDMENT.md")==
            read_json(PGROOT/"FIXED_SEVEN_SECONDARY_PROVENANCE.json")["reference_hashes"]["postrun_geometry/SCRATCH_MEAN_SECONDARY_AMENDMENT.md"],
            "scratch mean amendment changed")
    provenance,old,pins,rows=static_pins()
    five,F5=load_five(provenance,rows)
    fits=scratch_terminal_pair(SCRATCH)
    for arm in ("native","mto"):
        fit=fits[arm]["receipt"]
        require(fit["source_hashes"]==review["source_hashes"] and
                fit["code_hashes"]==review["code_hashes"],
                f"scratch {arm} receipt source mismatch")
    scratch={};evidence={}
    for arm in ("native","mto"):
        pred,details=selected_scratch_artifact(SCRATCH/f"runs/{arm}",
                                                 fits[arm]["receipt"],rows,arm)
        scratch[arm]=pred
        evidence[arm]={"terminal_sha256":fits[arm]["terminal_sha256"],**details}
    native=scratch["native"];mto=scratch["mto"]
    mean=(native+mto)/2
    group_rows,group_keys=identity_groups(rows)
    comparisons={name:compare(rows,F5,p,group_rows,group_keys)
                 for name,p in (("native_vs_F5_supplemental",native),
                                ("mto_vs_F5_supplemental",mto),
                                ("scratch_equal_mean_vs_F5_supplemental",mean))}
    result={"classification":"supplemental F5 scratch reference only; primary native-minus-MTO unchanged",
            "mode":"scratch family completed","time":time.time(),
            "protocol_sha256":sha(HERE/"PROTOCOL.md"),"script_sha256":sha(__file__),
            "review_sha256":sha(HERE/"IMPLEMENTATION_REVIEW.json"),
            "scratch_source_pins_sha256":SCRATCH_PINS_SHA,
            "scratch_terminal_and_selection_evidence":evidence,
            "inference_forward_counts":{"F5":5,"scratch_native":1,"scratch_mto":1,"scratch_equal_mean":2},
            "metrics":{k:metric(rows["truth"],v) for k,v in
                (("F5",F5),("scratch_native_raw_f_selected",native),
                 ("scratch_mto_raw_f_selected",mto),("scratch_equal_mean_secondary",mean))},
            "comparisons":comparisons,
            "existing_primary_native_minus_mto_delta_sse":
                metric(rows["truth"],native)["sse"]-metric(rows["truth"],mto)["sse"],
            "existing_primary_native_minus_mto_delta_r2":
                metric(rows["truth"],native)["r2"]-metric(rows["truth"],mto)["r2"],
            "fixed_q2_boundaries_train_only":rows["bounds2"].tolist(),
            "fixed_tail_thresholds":{"q90":Q90,"q99":Q99},
            "limitations":["F5 is supplemental and exploratory; it does not alter scratch primary native-minus-MTO.",
                           "No seed completion required; no scratch model inference or test.",
                           "Forward counts are not calibrated latency ratios."]}
    write_json(HERE/"SCRATCH_F5_SUPPLEMENTAL_RESULTS.json",result)
    report(result,HERE/"SCRATCH_F5_SUPPLEMENTAL_REPORT.md")
    print(json.dumps({"supplemental_complete":True,"comparisons":len(comparisons)}))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    options=ap.add_mutually_exclusive_group(required=True)
    options.add_argument("--fixture",action="store_true")
    options.add_argument("--seed-completed",action="store_true")
    options.add_argument("--scratch-supplemental",action="store_true")
    args=ap.parse_args()
    if args.fixture:fixture()
    elif args.seed_completed:completed_seed()
    else:scratch_supplemental()
