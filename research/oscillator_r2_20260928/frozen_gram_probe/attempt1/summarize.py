#!/usr/bin/env python3
"""Checkpoint-free fixed Gram probe evaluation after FIT_FROZEN."""
import json,math,sys,time
from pathlib import Path
import numpy as np
from features import ROOT,RESEARCH,FROZEN,ARCH,SRC,sha,save_json,cfg,load_existing_cache
from moments import row_matrix
def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    d=p-y;sse=float(np.square(d).sum());sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":sse,"sst":sst,"r2":float(1-sse/sst),
        "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
def atomic_npz(path,**arrays):
    p=Path(path);tmp=p.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays);tmp.replace(p)
def diagnostics(cache,signed,projected,part,c):
    y=cache["f_true"];base=cache["base64"]
    assert signed.shape==projected.shape==y.shape
    assert np.array_equal(projected,np.maximum(0,signed))
    mse_signed=float(np.square(signed-y).sum());mse_projected=float(np.square(projected-y).sum())
    assert mse_projected<=mse_signed+1e-9
    mask=signed<0
    energy=metric(cache["E_true"],cache["E_pred"])
    err=projected-y
    result={"signed":metric(y,signed),"projected":metric(y,projected),
        "frozen_base":metric(y,base),"energy":energy,
        "negative_signed_count":int(mask.sum()),"zero_signed_count":int((signed==0).sum()),
        "projection_sse_improvement":mse_signed-mse_projected,
        "mean_signed_prediction_error":float((signed-y).mean()),
        "signed_error_quantiles":np.quantile(signed-y,[0,.01,.5,.95,.99,1]).tolist(),
        "absolute_error_quantiles":np.quantile(np.abs(err),[0,.5,.9,.99,1]).tolist(),
        "max_projected_f":float(projected.max()),
        "per_state":[{"physical_state":j+1,**metric(y[:,j],projected[:,j])} for j in range(10)],
        "tail":{},"molecule_error_concentration":{}}
    for label,threshold in (("q90",c["validation_q90"]),("q99",c["validation_q99"])):
        sections={}
        for group,m in (("bright",y>=threshold),("below",y<threshold)):
            sections[group]={"count":int(m.sum()),"sse":float(np.square(projected[m]-y[m]).sum()),
                "base_sse":float(np.square(base[m]-y[m]).sum())}
        result["tail"][label]={"threshold":threshold,"subsets":sections}
    per_mol=np.square(projected-y).sum(axis=1)
    order=np.argsort(-per_mol,kind="stable")
    for name,k in (("top1",1),("top5",5),("top10",10),
                   ("top1pct",max(1,math.ceil(.01*len(y))))):
        result["molecule_error_concentration"][name]={"molecules":k,
            "share_of_sse":float(per_mol[order[:k]].sum()/per_mol.sum())}
    matches=np.flatnonzero(cache["ids"]==14562)
    if part=="val":
        assert len(matches)==1
        i=int(matches[0])
        result["fixed_prior_case14562"]={"id":14562,"global_index":int(cache["indices"][i]),
            "raw_true_f":y[i].tolist(),"frozen_base_f":base[i].tolist(),
            "signed_f":signed[i].tolist(),"projected_f":projected[i].tolist(),
            "molecule_sse":float(per_mol[i]),
            "frozen_base_molecule_sse":float(np.square(base[i]-y[i]).sum())}
    return result
def predict_all(cache,gram,stats,w,c):
    n=len(cache["ids"]);signed=[];projected=[];residual=[]
    for a in range(0,n,c["pred_batch_molecules"]):
        b=min(a+c["pred_batch_molecules"],n)
        X=row_matrix(cache,gram,a,b)
        z=(X-stats["mean"])/stats["std"]
        rhat=stats["mean_r"]+z[:,:len(w)]@w
        base=cache["base64"][a:b].reshape(-1)
        q=base+c["f_std_train"]*rhat
        signed.append(q.reshape(b-a,10))
        projected.append(np.maximum(0,q).reshape(b-a,10))
        residual.append(rhat.reshape(b-a,10))
    return np.concatenate(signed),np.concatenate(projected),np.concatenate(residual)
def evaluate_and_report(gpu,baseline,health,source,code,fit,train_manifest,val_manifest):
    c=cfg();t0=time.monotonic()
    assert (ROOT/"FIT_FROZEN.json").exists()
    assert sha(ROOT/"FIT_FROZEN.json")
    assert fit["source_hashes"]==source and fit["code_hashes"]==code
    runtime=ROOT/"runtime"
    for name,digest in fit["runtime_hashes"].items():assert sha(runtime/name)==digest
    with np.load(runtime/"training_statistics.npz",allow_pickle=False) as z:
        stats={k:z[k].copy() for k in z.files}
    stats["mean_r"]=float(stats["mean_r"])
    with np.load(runtime/"frozen_coefficients.npz",allow_pickle=False) as z:
        w={"A":z["w_A"].copy(),"B":z["w_B"].copy()}
        assert float(z["intercept_A"])==float(z["intercept_B"])==stats["mean_r"]
    assert len(w["A"])==129 and len(w["B"])==657
    traincache=load_existing_cache("train");valcache=load_existing_cache("val")
    grams={"train":np.load(runtime/"train_gram.npy",mmap_mode="r",allow_pickle=False),
           "val":np.load(runtime/"val_gram.npy",mmap_mode="r",allow_pickle=False)}
    assert sha(runtime/"train_gram.npy")==train_manifest["gram_sha256"]
    assert sha(runtime/"val_gram.npy")==val_manifest["gram_sha256"]
    for name,cache in (("train",traincache),("val",valcache)):
        assert cache["mask_f"].all() and cache["mask_E"].all()
        assert grams[name].shape==(len(cache["ids"]),10,528)
    results={}
    valpred={}
    for arm in ("A","B"):
        results[arm]={}
        for part,cache in (("train",traincache),("val",valcache)):
            signed,projected,resid=predict_all(cache,grams[part],stats,w[arm],c)
            result=diagnostics(cache,signed,projected,part,c)
            if part=="train":
                target=(cache["f_true"]-cache["base64"])/c["f_std_train"]
                mse=float(np.square(target-resid).mean())
                J=mse+c["ridge_lambda_mean_mse"]*float(np.dot(w[arm],w[arm]))
                assert abs(J-fit["solver"][arm]["objective"])<1e-8,(arm,J,fit["solver"][arm]["objective"])
                result["signed_ridge_objective_recomputed"]=J
            else:
                valpred[arm]=projected
                atomic_npz(runtime/f"val_predictions_{arm}.npz",
                    ids=cache["ids"],indices=cache["indices"],
                    f_true=cache["f_true"],E_true=cache["E_true"],E_pred=cache["E_pred"],
                    frozen_base_native64=cache["base64"],
                    signed_f=signed,projected_f=projected,linear_residual=resid)
            results[arm][part]=result
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        ids=z["ids"].copy();indices=z["indices"].copy();truth=z["f_true"].copy()
        assert np.array_equal(ids,valcache["ids"]) and np.array_equal(indices,valcache["indices"])
        assert np.array_equal(truth,valcache["f_true"])
        eta0=z["f"].copy()
    ensemble=[]
    ensemble_hashes={}
    for name in ("mto_eta0","mto_eta01","mto_eta1"):
        path=SRC/"runs"/name/"val_predictions.npz"
        ensemble_hashes[name]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z["ids"],ids) and np.array_equal(z["indices"],indices)
            assert np.array_equal(z["f_true"],truth)
            ensemble.append(z["f"].copy())
    equal3=sum(ensemble)/3
    layout=json.loads((SRC/"data/identity_audit_v2.json").read_text())
    groups=[]
    for ix,id_ in zip(indices,ids):
        item=layout[int(ix)]
        assert int(item[0])==int(id_) and item[2] is True
        groups.append(item[1])
    sys.path.insert(0,str(RESEARCH/"analysis_addendum"))
    from validation_comparison_addendum import bootstraps
    boot_A,group_layout=bootstraps(truth,eta0,{"A":valpred["A"],"B":valpred["B"]},groups)
    boot_BA,layout2=bootstraps(truth,valpred["A"],{"B_minus_A":valpred["B"]},groups)
    boot_eq,layout3=bootstraps(truth,equal3,{"A_vs_equal3":valpred["A"],"B_vs_equal3":valpred["B"]},groups)
    assert group_layout==layout2==layout3
    comparator={"eta0":metric(truth,eta0),"equal_three_native_f_mean":metric(truth,equal3)}
    diffs={"B_minus_A":results["B"]["val"]["projected"]["r2"]-results["A"]["val"]["projected"]["r2"],
        "A_minus_eta0":results["A"]["val"]["projected"]["r2"]-comparator["eta0"]["r2"],
        "B_minus_eta0":results["B"]["val"]["projected"]["r2"]-comparator["eta0"]["r2"],
        "A_minus_equal3":results["A"]["val"]["projected"]["r2"]-comparator["equal_three_native_f_mean"]["r2"],
        "B_minus_equal3":results["B"]["val"]["projected"]["r2"]-comparator["equal_three_native_f_mean"]["r2"]}
    after=health.snapshot(gpu)
    assert health.unchanged(baseline,after) and set(after["apps"]).issubset({__import__("os").getpid()})
    report={"classification":"One fixed-λ train-only two-arm convex probe, then one validation evaluation",
        "created":time.time(),"protocol_sha256":c["protocol_sha256"],"source_hashes":source,
        "code_hashes":code,"fit_frozen_sha256":sha(ROOT/"FIT_FROZEN.json"),
        "train_gram_manifest_sha256":sha(ROOT/"TRAIN_GRAM_MANIFEST.json"),
        "val_gram_manifest_sha256":sha(ROOT/"VAL_GRAM_MANIFEST.json"),
        "validation_predictions_server_only_hashes":{arm:sha(runtime/f"val_predictions_{arm}.npz") for arm in ("A","B")},
        "frozen_comparators_sha256":ensemble_hashes,
        "arms":results,"comparators":comparator,"delta_r2":diffs,
        "paired_bootstrap":{"vs_eta0":boot_A,"B_minus_A":boot_BA,"vs_equal_three":boot_eq},
        "connectivity_layout":group_layout,"fit_diagnostics":fit["solver"],
        "health_before":baseline,"health_after":after,
        "evaluation_seconds":time.monotonic()-t0,
        "decision_thresholds":{"candidate_gain_vs_eta0":.01,"Gram_gain_B_minus_A":.005},
        "limitations":["Validation split and feature design are historically exposed.",
            "Arm B has more coefficients; B-A is not a capacity-matched information-only theorem.",
            "Ridge minimizes signed residual error; nonnegative projection is a common post-fit rule.",
            "No validation-driven fit, alternative lambda, test evaluation or model promotion occurred."]}
    save_json(report,ROOT/"RESULTS.json")
    lines=["# Frozen invariant Gram probe","",
        "Two fixed train-only ridge fits with λ=0.001; coefficients frozen before validation Gram and predictions.",
        "",
        "| Predictor | Validation raw-f R² | ΔR² vs eta0 | Validation RMSE |",
        "| --- | ---: | ---: | ---: |"]
    lines.append(f"| eta0 frozen | {comparator['eta0']['r2']:.6f} | 0 | {comparator['eta0']['rmse']:.6f} |")
    for arm in ("A","B"):
        m=results[arm]["val"]["projected"]
        lines.append(f"| {arm} | {m['r2']:.6f} | {diffs[arm+'_minus_eta0']:+.6f} | {m['rmse']:.6f} |")
    lines.append(f"| fixed equal-three | {comparator['equal_three_native_f_mean']['r2']:.6f} | "+
        f"{comparator['equal_three_native_f_mean']['r2']-comparator['eta0']['r2']:+.6f} | "+
        f"{comparator['equal_three_native_f_mean']['rmse']:.6f} |")
    lines.extend(["",f"Primary B−A ΔR²: {diffs['B_minus_A']:+.6f}.",
        "Full train/validation state, tail, signed/projection and paired bootstrap results are in RESULTS.json.",
        "No test data or validation-driven coefficient changes were used."])
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    return report

