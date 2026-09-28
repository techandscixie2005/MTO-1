#!/usr/bin/env python3
"""Audit completed Gram computation after ASCII Markdown failure; no model inference."""
import fcntl,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
FROZEN=ROOT.parent/"frozen_residual"
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))
def metric(y,p):
    d=np.asarray(p,np.float64)-np.asarray(y,np.float64)
    y=np.asarray(y,np.float64)
    sse=float(np.square(d).sum());sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":sse,"sst":sst,"r2":1-sse/sst,
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
def same_metric(a,b):
    assert a["count"]==b["count"]
    for k in ("sse","sst","r2","mae","rmse"):
        assert math.isclose(float(a[k]),float(b[k]),rel_tol=1e-12,abs_tol=1e-12),(k,a[k],b[k])
def main():
    assert not (ROOT/"PROBE_COMPLETE.json").exists()
    assert not (ROOT/"POSTPROCESS_RECOVERY.json").exists()
    post_review=read("POSTPROCESS_REVIEW.json")
    assert post_review["passed"] and post_review["script_sha256"]==sha(__file__)
    assert post_review["results_sha256"]==sha(ROOT/"RESULTS.json")
    assert post_review["original_failed_sha256"]==sha(ROOT/"FAILED.json")
    failed=read("FAILED.json")
    assert failed["event"]=="FAILED" and failed["type"]=="UnicodeEncodeError"
    assert "'ascii' codec" in failed["error"]
    receipt=read("LAUNCH_RECEIPT.json")
    assert receipt["gpu"]==2
    proc=Path(f"/proc/{receipt['pid']}/stat")
    if proc.exists():
        fields=proc.read_text().split()
        assert int(fields[21])!=receipt["start_ticks"] or fields[2]=="Z"
    lock=open("/tmp/mto_pouter_gpu_2.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    pre=read("PREFLIGHT.json");review=read("IMPLEMENTATION_REVIEW.json")
    train=read("TRAIN_GRAM_MANIFEST.json");fit=read("FIT_FROZEN.json")
    val=read("VAL_GRAM_MANIFEST.json");result=read("RESULTS.json")
    assert pre["passed"] and review["passed"]
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    source=review["source_hashes"];code=review["code_hashes"]
    assert pre["source_hashes"]==source and pre["code_hashes"]==code
    for path,digest in {**source,**code}.items():assert sha(path)==digest,(path,"source changed")
    for obj in (receipt,train,fit,val,result):
        assert obj["source_hashes"]==source and obj["code_hashes"]==code
    assert receipt["review_sha256"]==sha(ROOT/"IMPLEMENTATION_REVIEW.json")
    assert receipt["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    assert receipt["started"]<train["created"]<fit["created"]<val["created"]<result["created"]<failed["time"]
    assert fit["event"]=="FIT_FROZEN" and fit["training_rows"]==1203550 and fit["lambda"]==.001
    assert result["fit_frozen_sha256"]==sha(ROOT/"FIT_FROZEN.json")
    assert fit["train_gram_manifest_sha256"]==sha(ROOT/"TRAIN_GRAM_MANIFEST.json")
    assert result["train_gram_manifest_sha256"]==sha(ROOT/"TRAIN_GRAM_MANIFEST.json")
    assert result["val_gram_manifest_sha256"]==sha(ROOT/"VAL_GRAM_MANIFEST.json")
    assert train["gram_sha256"]==sha(ROOT/"runtime/train_gram.npy")
    assert val["gram_sha256"]==sha(ROOT/"runtime/val_gram.npy")
    for name,digest in fit["runtime_hashes"].items():
        assert sha(ROOT/"runtime"/name)==digest
    with np.load(ROOT/"runtime/training_statistics.npz",allow_pickle=False) as z:
        stats={k:z[k].copy() for k in z.files}
    with np.load(ROOT/"runtime/frozen_coefficients.npz",allow_pickle=False) as z:
        weights={"A":z["w_A"].copy(),"B":z["w_B"].copy()}
        assert float(z["intercept_A"])==float(z["intercept_B"])==float(stats["mean_r"])
    solver_replay={}
    for arm,ncols in (("A",129),("B",657)):
        active=stats["active"][:ncols]
        ix=np.flatnonzero(active)
        C=stats["C"][np.ix_(ix,ix)]
        c=stats["c"][ix]
        w=weights[arm][ix]
        assert len(weights[arm])==ncols
        assert np.count_nonzero(weights[arm][~active])==0
        lam=.001
        residual=float(np.linalg.norm((C+lam*np.eye(len(ix)))@w-c)/max(float(np.linalg.norm(c)),1e-12))
        assert residual<=1e-10
        J=float(stats["r_variance"]-2*w@c+w@C@w+lam*w@w)
        assert math.isclose(J,fit["solver"][arm]["objective"],rel_tol=1e-12,abs_tol=1e-12)
        solver_replay[arm]={"normal_residual":residual,"objective":J,"active_columns":len(ix)}
    assert solver_replay["B"]["objective"]<=solver_replay["A"]["objective"]+1e-10
    with np.load(FROZEN/"runtime/cache_val.npz",allow_pickle=False) as z:
        ids=z["ids"].copy();indices=z["indices"].copy();truth=z["f_true"].copy()
        base=z["base64"].copy()
    recalc={}
    for arm in ("A","B"):
        path=ROOT/"runtime"/f"val_predictions_{arm}.npz"
        assert sha(path)==result["validation_predictions_server_only_hashes"][arm]
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z["ids"],ids) and np.array_equal(z["indices"],indices)
            assert np.array_equal(z["f_true"],truth)
            assert np.array_equal(z["frozen_base_native64"],base)
            signed=z["signed_f"].copy();pred=z["projected_f"].copy()
            assert np.array_equal(pred,np.maximum(0,signed))
            assert np.isfinite(pred).all()
        recalc[arm]=metric(truth,pred)
        same_metric(recalc[arm],result["arms"][arm]["val"]["projected"])
    delta=recalc["B"]["r2"]-recalc["A"]["r2"]
    assert math.isclose(delta,result["delta_r2"]["B_minus_A"],rel_tol=0,abs_tol=1e-12)
    assert result["health_before"]["ecc"]==result["health_after"]["ecc"]
    assert receipt["gpu_uuid"]==result["health_before"]["uuid"]==result["health_after"]["uuid"]
    for method in ("molecule","connectivity_group"):
        row=result["paired_bootstrap"]["B_minus_A"][method]["B_minus_A"]
        assert row["replicates"]==2000 and row["seed"]==20260928
        assert all(np.isfinite(row["delta_r2_2p5_50_97p5"]))
    report_lines=["# Frozen invariant Gram probe: recovered report","",
        "Scientific fit and validation completed. The original worker failed while writing Markdown under an ASCII locale; no model inference or fitting was repeated for this recovery.","",
        "| Predictor | Validation raw-f R2 | Difference versus eta0 | RMSE |",
        "| --- | ---: | ---: | ---: |"]
    comparator=result["comparators"]
    eta=comparator["eta0"]["r2"]
    report_lines.append(f"| eta0 | {eta:.6f} | 0 | {comparator['eta0']['rmse']:.6f} |")
    for arm in ("A","B"):
        m=recalc[arm]
        report_lines.append(f"| {arm} | {m['r2']:.6f} | {m['r2']-eta:+.6f} | {m['rmse']:.6f} |")
    eq=comparator["equal_three_native_f_mean"]
    report_lines.append(f"| fixed equal-three | {eq['r2']:.6f} | {eq['r2']-eta:+.6f} | {eq['rmse']:.6f} |")
    report_lines.extend(["",f"B minus A validation R2: {delta:+.6f}.",
        f"Train-only ridge objective: A {solver_replay['A']['objective']:.9f}; B {solver_replay['B']['objective']:.9f}.",
        "The extra Gram coordinates did not improve validation R2 in this fixed linear probe.",
        "Validation is historically exposed. No test access, refit, tuning or additional model inference occurred.",
        "The original FAILED.json and absent PROBE_COMPLETE.json remain part of the provenance."])
    md=ROOT/"RESULTS_RECOVERED.md"
    md.write_text("\n".join(report_lines)+"\n",encoding="utf-8")
    outcome={"classification":"Scientific fit and validation completed; original worker failed only during ASCII Markdown report write; additive reviewed recovery",
        "time":time.time(),"original_failed_sha256":sha(ROOT/"FAILED.json"),
        "original_worker_receipt_sha256":sha(ROOT/"LAUNCH_RECEIPT.json"),
        "source_hashes":source,"code_hashes":code,
        "postprocess_source_sha256":sha(__file__),
        "postprocess_review_sha256":sha(ROOT/"POSTPROCESS_REVIEW.json"),
        "report_utf8_sha256":sha(md),"results_sha256":sha(ROOT/"RESULTS.json"),
        "fit_frozen_sha256":sha(ROOT/"FIT_FROZEN.json"),
        "train_gram_manifest_sha256":sha(ROOT/"TRAIN_GRAM_MANIFEST.json"),
        "val_gram_manifest_sha256":sha(ROOT/"VAL_GRAM_MANIFEST.json"),
        "server_only_array_hashes":result["validation_predictions_server_only_hashes"],
        "solver_replay":solver_replay,"validation_recomputed":recalc,
        "delta_B_minus_A":delta,"phase_order_verified":True,
        "worker_exited_and_gpu_lock_free":True,
        "no_refit_no_model_inference_no_test":True,
        "original_PROBE_COMPLETE_absent":True}
    path=ROOT/"POSTPROCESS_RECOVERY.json"
    path.write_text(json.dumps(outcome,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"recovered":True,"A_R2":recalc["A"]["r2"],
        "B_R2":recalc["B"]["r2"],"receipt":str(path)}))
if __name__=="__main__":main()

