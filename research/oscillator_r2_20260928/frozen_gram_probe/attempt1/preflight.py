#!/usr/bin/env python3
"""Train-only and synthetic preflight; never build full Gram or fit full data."""
import fcntl,json,math,os,sys,time
import numpy as np
import torch
from features import ROOT,ARCH,FROZEN,SRC,Split,source_model,setup_torch,sha,save_json,cfg,source_seal,code_hashes,upper_index_map,compute_gram,live_features,load_existing_cache,verify_alignment,compare_cached
from moments import Moments,solve_both
def assertclose(a,b,rtol=1e-5,atol=1e-6,name="value"):
    if not torch.allclose(a,b,rtol=rtol,atol=atol):
        raise AssertionError(f"{name} maxdiff={float((a-b).abs().max())}")
def main():
    c=cfg();gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu)
    assert gpu in c["allowed_gpu_clean"] and torch.cuda.device_count()==1
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH))
    import gpu_health
    before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
    setup_torch(11)
    source=source_seal();code=code_hashes()
    assert len(upper_index_map())==528
    assert [(e["k"],e["l"]) for e in upper_index_map()]==list(zip(*np.triu_indices(32)))
    split=Split("train")
    cache=load_existing_cache("train")
    verify_alignment("train",split,cache)
    assert len(split.indices)==120355 and not np.intersect1d(
        split.indices,np.load(SRC/"data/dataset.npz",allow_pickle=False)["val"]).size
    model=source_model("cuda").eval()
    assert all(p.requires_grad is False for p in model.base.parameters())
    x=split.batch(0,64,"cuda")
    live=live_features(model,x)
    compare_cached(cache,0,64,live)
    assert live["T"].shape==(64,10,32,5)
    assert live["gram"].shape==(64,10,528) and live["gram"].dtype==torch.float64
    assertclose(live["trace_gram"],live["trace_source"],rtol=1e-5,atol=2e-5,name="trace")
    basis=model.base.decoder.cartesian_basis
    basis_gram=basis.double().reshape(5,9)@basis.double().reshape(5,9).T
    max_basis_orthonormal_error=float((basis_gram-torch.eye(5,device="cuda")).abs().max())
    # No approximation by T@T is used, even though this finite-precision basis is near orthonormal.
    symmetric=float((live["gram_full"]-live["gram_full"].transpose(-1,-2)).abs().max())
    min_eig=float(torch.linalg.eigvalsh(live["gram_full"][:2]).min())
    assert symmetric<1e-12 and min_eig>=-1e-10
    vecnorm=float(live["gram"][:2].square().sum())
    frobenius=float(live["gram_full"][:2].square().sum())
    assert abs(vecnorm-frobenius)<=1e-9*max(1.,frobenius)
    # Cartesian tensor rotations leave the Gram unchanged.
    theta=.37
    R=torch.tensor([[math.cos(theta),-math.sin(theta),0],
                    [math.sin(theta),math.cos(theta),0],[0,0,1]],dtype=torch.float64,device="cuda")
    V=live["V"][:2]
    rotated=torch.einsum("ia,...kab,jb->...kij",R,V,R)
    f=rotated.flatten(-2);rotgram=f@f.transpose(-1,-2)
    assertclose(rotgram,live["gram_full"][:2],rtol=1e-10,atol=1e-10,name="synthetic Cartesian rotation")
    # A real fixed train example is also invariant when the input coordinates rotate.
    xr={k:v for k,v in x.items()}
    xr["pos"]=x["pos"]@R.float().T
    rotated_source=live_features(model,xr)
    source_rotation_max=float((rotated_source["gram"]-live["gram"]).abs().max())
    assertclose(rotated_source["gram"],live["gram"],rtol=1e-4,atol=1e-5,name="source rotation")
    # Within-molecule atom reordering, with edges relabeled, cannot change state features.
    atoms=len(x["z"]);batch=x["batch"]
    perm=torch.cat([torch.nonzero(batch==j).flatten().flip(0) for j in range(64)])
    inverse=torch.empty_like(perm);inverse[perm]=torch.arange(atoms,device="cuda")
    xp={"z":x["z"][perm],"pos":x["pos"][perm],"batch":batch[perm],
        "n":64,"edge_index":inverse[x["edge_index"]]}
    permuted=live_features(model,xp)
    source_permutation_max=float((permuted["gram"]-live["gram"]).abs().max())
    assertclose(permuted["gram"],live["gram"],rtol=1e-4,atol=1e-5,name="source atom permutation")
    # Fixed first 16 TRAIN molecules: block-merged moments versus direct dense center.
    h=cache["h"][:16].reshape(-1,128).astype(np.float64)
    base=cache["base64"][:16].reshape(-1,1)
    g=live["gram"][:16].cpu().numpy().reshape(-1,528)
    X=np.concatenate((h,base,g),axis=1)
    r=((cache["f_true"][:16]-cache["base64"][:16])/c["f_std_train"]).reshape(-1)
    moments=Moments(657)
    for a in range(0,len(r),17):moments.add(X[a:a+17],r[a:a+17],device="cpu")
    stats=moments.finalize()
    assert stats["n"]==160
    assert np.allclose(stats["mean"],X.mean(0),rtol=1e-12,atol=1e-12)
    assert abs(stats["mean_r"]-float(r.mean()))<1e-12
    z=(X-stats["mean"])/stats["std"]
    z[:,~stats["active"]]=0
    C=z.T@z/len(r);cross=z.T@(r-r.mean())/len(r)
    covariance_max=float(np.max(np.abs(C-stats["C"])))
    cross_max=float(np.max(np.abs(cross-stats["c"])))
    assert np.allclose(C,stats["C"],rtol=1e-10,atol=1e-10)
    assert np.allclose(cross,stats["c"],rtol=1e-10,atol=1e-10)
    cuda_moments=Moments(657)
    for a in range(0,len(r),17):cuda_moments.add(X[a:a+17],r[a:a+17],device="cuda")
    cuda_stats=cuda_moments.finalize()
    cuda_moment_max=float(max(np.max(np.abs(cuda_stats["mean"]-stats["mean"])),
        np.max(np.abs(cuda_stats["C"]-stats["C"])),
        np.max(np.abs(cuda_stats["c"]-stats["c"]))))
    assert cuda_stats["n"]==stats["n"] and np.array_equal(cuda_stats["active"],stats["active"])
    assert np.allclose(cuda_stats["mean"],stats["mean"],rtol=1e-10,atol=1e-11)
    assert np.allclose(cuda_stats["C"],stats["C"],rtol=1e-10,atol=1e-10)
    assert np.allclose(cuda_stats["c"],stats["c"],rtol=1e-10,atol=1e-10)
    solved=solve_both(stats,c["ridge_lambda_mean_mse"])
    cuda_solved=solve_both(cuda_stats,c["ridge_lambda_mean_mse"])
    for arm in ("A","B"):
        assert np.allclose(cuda_solved[arm]["w"],solved[arm]["w"],rtol=1e-9,atol=1e-10)
    for arm in ("A","B"):
        row=solved[arm];cols=row["ncols"]
        idx=row["active_indices"]
        A=C[np.ix_(idx,idx)]+c["ridge_lambda_mean_mse"]*np.eye(len(idx))
        direct=np.linalg.solve(A,cross[idx])
        assert np.allclose(row["w"][idx],direct,rtol=1e-9,atol=1e-10)
    # Synthetic nonnegative truth: fixed max(0,signed) never increases SSE.
    truth=np.array([0.,.1,1.,2.])
    signed=np.array([-.2,-.1,.7,2.3])
    assert np.square(np.maximum(0,signed)-truth).sum()<=np.square(signed-truth).sum()
    after=gpu_health.snapshot(gpu)
    assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
    report={"passed":True,"classification":"Synthetic, first64 train feature invariance and first16 train ridge subset preflight only; no full fit, no validation inference",
        "created":time.time(),"source_hashes":source,"code_hashes":code,
        "physical_gpu":gpu,"gpu_before":before,"gpu_after":after,
        "feature_dimensions":{"T":[64,10,32,5],"gram":[64,10,528],"common_slopes":129,"full_slopes":657},
        "upper_index_map":upper_index_map(),
        "basis_max_orthonormal_error":max_basis_orthonormal_error,
        "Gram_symmetry_max":symmetric,"Gram_min_eigenvalue_first2":min_eig,
        "Frobenius_upper_vector_identity":True,
        "source_trace_reconstruction":True,
        "source_rotation_gram_max_difference":source_rotation_max,
        "source_atom_permutation_gram_max_difference":source_permutation_max,
        "first64_frozen_cache_match":True,
        "train_subset_dense_stream_covariance_max_difference":covariance_max,
        "train_subset_dense_stream_cross_max_difference":cross_max,
        "train_subset_ridge_direct_solve_match":True,
        "train_subset_cuda_cpu_moments_max_difference":cuda_moment_max,
        "train_subset_cuda_cpu_ridge_match":True,
        "common_principal_block_same_stats":True,
        "synthetic_nonnegative_projection_sse_inequality":True,
        "no_val_gram_or_full_fit_or_test_access":True}
    save_json(report,ROOT/"PREFLIGHT.json")
    print(json.dumps({"passed":True,"rotation_max":source_rotation_max,
        "permutation_max":source_permutation_max,
        "covariance_max":covariance_max,"source_count":len(source),
        "code_count":len(code)}))
if __name__=="__main__":main()

