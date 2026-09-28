#!/usr/bin/env python3
"""One reviewed two-arm frozen eta0 Gram probe. Train fit freezes before validation."""
import fcntl,json,math,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from features import ROOT,RESEARCH,FROZEN,ARCH,SRC,Split,source_model,setup_torch,C_F,sha,save_json,cfg,source_seal,code_hashes,upper_index_map,live_features,load_existing_cache,verify_alignment,compare_cached
from moments import Moments,row_matrix,residual,solve_both
STOP=False
def stop_requested(*_):
    global STOP
    STOP=True
def atomic_npz(path,**arrays):
    p=Path(path);tmp=p.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays);os.replace(tmp,p)
def health_guard(index,baseline,epoch,from_gpu_health):
    now=from_gpu_health.snapshot(index)
    if not from_gpu_health.unchanged(baseline,now) or not set(now["apps"]).issubset({os.getpid()}):
        save_json({"event":"INVALID_GPU_HEALTH_OR_OCCUPANCY","phase":epoch,
            "before":baseline,"observed":now,"time":time.time()},ROOT/"INVALID.json")
        raise RuntimeError("GPU health or ownership changed")
    return now
def build_gram(part,model,gpu,baseline,health,source,code):
    assert part in ("train","val")
    c=cfg();runtime=ROOT/"runtime";runtime.mkdir(exist_ok=True)
    final=runtime/f"{part}_gram.npy"
    manifest=ROOT/(f"{part.upper()}_GRAM_MANIFEST.json")
    if manifest.exists():
        m=json.loads(manifest.read_text())
        assert final.exists() and m["gram_sha256"]==sha(final)
        assert sha(Path(m["ids_indices_path"]))==m["ids_indices_sha256"]
        assert m["source_hashes"]==source and m["code_hashes"]==code
        assert m["feature_order"]=="lex k<=l, off-diagonal sqrt(2)"
        return m
    if part=="val":assert (ROOT/"FIT_FROZEN.json").exists(),"Validation features require frozen fit"
    split=Split(part);cache=load_existing_cache(part)
    verify_alignment(part,split,cache)
    assert cache["mask_f"].all() and cache["mask_E"].all()
    n=len(split.indices)
    shape=(n,10,c["gram_coordinates"])
    tmp=runtime/f"{part}_gram.tmp.npy"
    progress=ROOT/(f"{part.upper()}_GRAM_PROGRESS.json")
    # A crash after the flushed memmap rename but before its manifest can be
    # recovered by replaying from the last committed progress cursor.
    if final.exists() and not manifest.exists():
        assert progress.exists() and not tmp.exists(),"Untracked completed Gram without manifest"
        os.replace(final,tmp)
    if progress.exists():
        p=json.loads(progress.read_text())
        assert tmp.exists() and p["source_hashes"]==source and p["code_hashes"]==code
        assert p["shape"]==list(shape)
        start=int(p["cursor"]);assert 0<=start<=n and (start==n or start%c["feature_batch_molecules"]==0)
        gram=np.lib.format.open_memmap(tmp,mode="r+")
        assert gram.shape==shape and gram.dtype==np.float64
    else:
        assert not tmp.exists(),"Untracked partial Gram file"
        start=0
        gram=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float64,shape=shape)
    model.eval();t0=time.monotonic();checks=0
    for a in range(start,n,c["feature_batch_molecules"]):
        b=min(a+c["feature_batch_molecules"],n)
        if a%(100*c["feature_batch_molecules"])==0:
            health_guard(gpu,baseline,f"{part}_gram_{a}",health)
        live=live_features(model,split.batch(a,b,"cuda"))
        compare_cached(cache,a,b,live)
        if not torch.allclose(live["trace_gram"],live["trace_source"],
                              rtol=1e-5,atol=2e-5):
            diff=float((live["trace_gram"]-live["trace_source"]).abs().max())
            raise AssertionError(f"Source trace/Gram reconstruction mismatch max={diff}")
        values=live["gram"].cpu().numpy()
        assert values.shape==(b-a,10,528) and np.isfinite(values).all()
        gram[a:b]=values;checks+=1
        if (a//c["feature_batch_molecules"]+1)%100==0 or STOP:
            gram.flush()
            save_json({"part":part,"cursor":b,"shape":list(shape),
                "source_hashes":source,"code_hashes":code,"gpu":gpu,
                "time":time.time()},progress)
            save_json({"event":"GRAM_FEATURES","part":part,"cursor":b,
                "molecules":n,"gpu":gpu,"time":time.time()},ROOT/"status.json")
        if STOP:
            save_json({"event":"STOPPED_CHECKPOINTED","part":part,
                "cursor":b,"time":time.time()},ROOT/"status.json")
            return None
    gram.flush();del gram
    health_guard(gpu,baseline,f"{part}_gram_complete",health)
    os.replace(tmp,final)
    ids=runtime/f"{part}_gram_ids.npz"
    atomic_npz(ids,ids=split.ids,indices=split.indices)
    m={"part":part,"created":time.time(),"shape":list(shape),"dtype":"float64",
        "gram_path":str(final),"gram_sha256":sha(final),"gram_bytes":final.stat().st_size,
        "ids_indices_path":str(ids),"ids_indices_sha256":sha(ids),
        "feature_order":"lex k<=l, off-diagonal sqrt(2)",
        "upper_index_map":upper_index_map(),"source_hashes":source,"code_hashes":code,
        "source_model_eval":True,"source_tensor_dtype":"float32",
        "cartesian_projection_dtype":"float64",
        "cached_h_E_base_compared_each_batch":True,
        "trace_reconstruction_compared_each_batch":True,
        "batches_checked":checks,"seconds_this_process":time.monotonic()-t0,
        "gpu_before":baseline,"gpu_after":health_guard(gpu,baseline,part,health)}
    save_json(m,manifest)
    if progress.exists():progress.unlink()
    save_json({"event":"GRAM_COMPLETE","part":part,"molecules":n,
        "gram_sha256":m["gram_sha256"],"time":time.time()},ROOT/"status.json")
    return m
def fit_train(gpu,baseline,health,source,code,train_manifest):
    marker=ROOT/"FIT_FROZEN.json"
    if marker.exists():
        fit=json.loads(marker.read_text())
        assert fit["source_hashes"]==source and fit["code_hashes"]==code
        assert fit["train_gram_manifest_sha256"]==sha(ROOT/"TRAIN_GRAM_MANIFEST.json")
        for name,digest in fit["runtime_hashes"].items():
            assert sha(ROOT/"runtime"/name)==digest
        return fit
    c=cfg()
    cache=load_existing_cache("train")
    gram=np.load(ROOT/"runtime/train_gram.npy",mmap_mode="r",allow_pickle=False)
    assert gram.shape==(c["train_molecules"],10,528) and gram.dtype==np.float64
    assert sha(ROOT/"runtime/train_gram.npy")==train_manifest["gram_sha256"]
    moments=Moments(c["expanded_columns"])
    t0=time.monotonic()
    for a in range(0,c["train_molecules"],c["moment_batch_molecules"]):
        b=min(a+c["moment_batch_molecules"],c["train_molecules"])
        if a%(10*c["moment_batch_molecules"])==0:
            health_guard(gpu,baseline,f"moments_{a}",health)
        X=row_matrix(cache,gram,a,b)
        r=residual(cache,a,b)
        moments.add(X,r,device="cuda")
        if STOP:
            save_json({"event":"STOPPED_BEFORE_FIT_FROZEN","phase":"moments",
                "cursor":b,"time":time.time()},ROOT/"status.json")
            return None
    stats=moments.finalize()
    assert stats["n"]==1203550 and stats["C"].shape==(657,657)
    assert np.allclose(stats["C"][:129,:129],
        stats["C"][np.ix_(np.arange(129),np.arange(129))],rtol=0,atol=0)
    solved=solve_both(stats,c["ridge_lambda_mean_mse"])
    if STOP:
        save_json({"event":"STOPPED_BEFORE_FIT_FROZEN","phase":"solver",
            "time":time.time()},ROOT/"status.json")
        return None
    runtime=ROOT/"runtime"
    statpath=runtime/"training_statistics.npz"
    coefpath=runtime/"frozen_coefficients.npz"
    atomic_npz(statpath,mean=stats["mean"],std=stats["std"],active=stats["active"],
        C=stats["C"],c=stats["c"],mean_r=np.array(stats["mean_r"]),
        r_variance=np.array(stats["r_variance"]),
        minimum=stats["minimum"],maximum=stats["maximum"],
        constant_indices=stats["constant_indices"],constant_values=stats["constant_values"])
    atomic_npz(coefpath,w_A=solved["A"]["w"],w_B=solved["B"]["w"],
        intercept_A=np.array(solved["A"]["intercept_centered"]),
        intercept_B=np.array(solved["B"]["intercept_centered"]))
    diagnostics={}
    for name in ("A","B"):
        row=solved[name]
        diagnostics[name]={k:v for k,v in row.items() if k not in ("w","active_indices")}
        diagnostics[name]["active_indices"]=row["active_indices"].tolist()
    fit={"event":"FIT_FROZEN","created":time.time(),
        "classification":"Train-only fixed-lambda ridge fits frozen before validation Gram/predictions",
        "source_hashes":source,"code_hashes":code,
        "train_gram_manifest_sha256":sha(ROOT/"TRAIN_GRAM_MANIFEST.json"),
        "runtime_hashes":{statpath.name:sha(statpath),coefpath.name:sha(coefpath)},
        "training_rows":stats["n"],"lambda":c["ridge_lambda_mean_mse"],
        "target":"(raw_f-base_native64)/train_f_std",
        "feature_order":"128 h, native64 base, 528 lex Cartesian upper Gram with sqrt2 off-diagonal",
        "intercept_unpenalized":True,
        "constant_columns":stats["constant_indices"].tolist(),
        "raw_moment_symmetry_relative":stats["raw_centered_symmetry_relative"],
        "common_stats_exact_principal_block":True,
        "J_B_le_J_A_verified":True,
        "solver":diagnostics,"moments_seconds":time.monotonic()-t0,
        "gpu_before":baseline,"gpu_after":health_guard(gpu,baseline,"fit_frozen",health),
        "no_validation_feature_or_target_loaded":True}
    save_json(fit,marker)
    save_json({"event":"FIT_FROZEN","time":time.time(),
        "A_objective":diagnostics["A"]["objective"],
        "B_objective":diagnostics["B"]["objective"]},ROOT/"status.json")
    return fit
def main():
    c=cfg();gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu)
    assert gpu in c["allowed_gpu_clean"] and torch.cuda.device_count()==1
    sys.path.insert(0,str(ARCH))
    import gpu_health
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own=open(ROOT/"worker.lock","a+")
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=gpu_health.snapshot(gpu)
    assert gpu_health.eligible(baseline)
    signal.signal(signal.SIGTERM,stop_requested)
    signal.signal(signal.SIGINT,stop_requested)
    setup_torch(11)
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and pre["source_hashes"]==source and pre["code_hashes"]==code
    assert review["passed"] and review["source_hashes"]==source and review["code_hashes"]==code
    assert not (ROOT/"PROBE_COMPLETE.json").exists()
    assert not (ROOT/"FAILED.json").exists() and not (ROOT/"INVALID.json").exists()
    receipt_path=ROOT/"LAUNCH_RECEIPT.json"
    for _ in range(100):
        if receipt_path.exists():break
        time.sleep(.1)
    assert receipt_path.exists(),"Launch receipt missing"
    receipt=json.loads(receipt_path.read_text())
    assert receipt["gpu"]==gpu and receipt["review_sha256"]==sha(ROOT/"IMPLEMENTATION_REVIEW.json")
    assert receipt["pid"]==os.getpid()
    assert receipt["start_ticks"]==int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21])
    assert receipt["cwd"]==str(ROOT) and Path.cwd()==ROOT
    assert receipt["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    assert receipt["code_hashes"]==code and receipt["source_hashes"]==source
    assert receipt["gpu_uuid"]==baseline["uuid"]
    model=source_model("cuda").eval()
    for p in model.parameters():p.requires_grad_(False)
    train=build_gram("train",model,gpu,baseline,gpu_health,source,code)
    if train is None:return
    del model;torch.cuda.empty_cache()
    fit=fit_train(gpu,baseline,gpu_health,source,code,train)
    if fit is None:return
    if STOP:return
    model=source_model("cuda").eval()
    for p in model.parameters():p.requires_grad_(False)
    val=build_gram("val",model,gpu,baseline,gpu_health,source,code)
    if val is None:return
    del model;torch.cuda.empty_cache()
    sys.path.insert(0,str(ROOT))
    from summarize import evaluate_and_report
    report=evaluate_and_report(gpu,baseline,gpu_health,source,code,fit,train,val)
    health_guard(gpu,baseline,"complete",gpu_health)
    save_json({"event":"PROBE_COMPLETE","created":time.time(),
        "source_hashes":source,"code_hashes":code,
        "fit_frozen_sha256":sha(ROOT/"FIT_FROZEN.json"),
        "results_sha256":sha(ROOT/"RESULTS.json"),
        "train_gram_sha256":train["gram_sha256"],"val_gram_sha256":val["gram_sha256"],
        "gpu_before":baseline,"gpu_after":health_guard(gpu,baseline,"complete",gpu_health)},ROOT/"PROBE_COMPLETE.json")
    print(json.dumps({"complete":True,"A_val_r2":report["arms"]["A"]["val"]["projected"]["r2"],
        "B_val_r2":report["arms"]["B"]["val"]["projected"]["r2"],
        "result":str(ROOT/"RESULTS.json")}),flush=True)
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        save_json({"event":"FAILED","type":type(exc).__name__,
            "error":str(exc),"time":time.time()},ROOT/"FAILED.json")
        raise

