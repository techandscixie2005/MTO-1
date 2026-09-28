#!/usr/bin/env python3
"""Frozen eta0 Cartesian tensor Gram features; no target enters a forward call."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
FROZEN=RESEARCH/"frozen_residual"
ARCH=RESEARCH/"architecture"
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(FROZEN))
from common import Split,source_model,setup_torch,C_F
TRIU=np.triu_indices(32)
OFFDIAG=TRIU[0]!=TRIU[1]
SQRT2=math.sqrt(2.0)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def save_json(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)
def cfg():
    return json.loads((ROOT/"config.json").read_text())
def source_seal():
    c=cfg()
    assert sha(ROOT/"FROZEN_GRAM_PROBE_PROTOCOL.md")==c["protocol_sha256"]
    assert sha(SRC/"runs/mto_eta0/best.pt")==c["source_eta0_checkpoint_sha256"]
    assert sha(ARCH/"initial/retained_residual.pt")==c["initial_retained_residual_checkpoint_sha256"]
    assert sha(FROZEN/"runtime/cache_train.npz")==c["cache_train_sha256"]
    assert sha(FROZEN/"runtime/cache_val.npz")==c["cache_val_sha256"]
    prior=json.loads((ARCH/"PREFLIGHT.json").read_text())
    audit=json.loads((ARCH/"INDEPENDENT_COMPLETION_AUDIT.json").read_text())
    frozen=json.loads((FROZEN/"INDEPENDENT_RUN_AUDIT.json").read_text())
    assert prior["passed"] and audit["passed"] and frozen["passed"]
    spread=json.loads((ROOT/"GRAM_SPREAD_TRAIN_GPU2.json").read_text())
    assert spread["passed"] and spread["gpu"]==2 and len(spread["starts"])==17
    assert spread["trace_mismatch_count"]==0
    assert all(item["failure_count"]==0 for item in spread["keys"].values())
    failed=json.loads((ROOT/"attempt1/FAILED.json").read_text())
    assert failed["event"]=="FAILED" and failed["type"]=="AssertionError"
    reset=json.loads((ROOT/"ATTEMPT1_RESET_RECEIPT.json").read_text())
    assert reset["no_train_manifest"] and reset["no_frozen_fit"] and reset["no_val_gram"]
    attempt=json.loads((ROOT/"attempt1/MANIFEST.json").read_text())
    assert attempt["failed_pre_fit"] and attempt["fit_frozen_absent"] and attempt["validation_gram_absent"]
    for p,h in prior["source_hashes"].items():assert sha(p)==h
    assert c["allow_test"] is False and c["allow_validation_refit"] is False
    for path,digest in c["pinned_direct_dependencies_sha256"].items():
        assert sha(path)==digest,(path,"pinned dependency changed")
    paths=[ROOT/"FROZEN_GRAM_PROBE_PROTOCOL.md",ROOT/"config.json",
        ROOT/"NUMERICAL_AMENDMENT.md",ROOT/"attempt1/MANIFEST.json",
        ROOT/"ATTEMPT1_RESET_RECEIPT.json",
        ROOT/"GRAM_SPREAD_TRAIN_GPU2.json",
        ROOT/"NUMERICAL_AMENDMENT_SCIENTIFIC_REVIEW.md",
        ARCH/"PREFLIGHT.json",ARCH/"INDEPENDENT_COMPLETION_AUDIT.json",
        FROZEN/"CACHE_MANIFEST.json",FROZEN/"INDEPENDENT_RUN_AUDIT.json",
        FROZEN/"runtime/cache_train.npz",FROZEN/"runtime/cache_val.npz"]
    paths.extend(Path(path) for path in c["pinned_direct_dependencies_sha256"])
    return {str(p):sha(p) for p in paths}
def code_hashes():
    files=["features.py","moments.py","preflight.py","run.py","summarize.py","launch.py","config.json","FROZEN_GRAM_PROBE_PROTOCOL.md"]
    return {str(ROOT/x):sha(ROOT/x) for x in files}
def upper_index_map():
    return [{"column":j,"k":int(k),"l":int(l),"multiplier":SQRT2 if k!=l else 1.0}
            for j,(k,l) in enumerate(zip(*TRIU))]
def compute_gram(T,basis):
    """Both original FP32 inputs are promoted before the Cartesian projection."""
    assert T.dtype==torch.float32 and basis.dtype==torch.float32
    assert T.shape[-2:]==(32,5) and basis.shape==(5,3,3)
    V=torch.einsum("...km,mij->...kij",T.double(),basis.double())
    flat=V.flatten(-2)
    full=flat@flat.transpose(-1,-2)
    upper=full[...,TRIU[0],TRIU[1]]
    scale=torch.as_tensor(np.where(OFFDIAG,SQRT2,1.0),dtype=torch.float64,device=T.device)
    return upper*scale,full,V
def live_features(model,x):
    with torch.inference_mode():
        h,T,E=model.features(**x)
        A=model.original_matrix(h,T)
        tr32=A.diagonal(dim1=-2,dim2=-1).sum(-1)
        base32=C_F*E*tr32
        base64=C_F*E.double()*A.double().diagonal(dim1=-2,dim2=-1).sum(-1)
        G,full,V=compute_gram(T,model.base.decoder.cartesian_basis)
        beta=model.base.decoder.beta_head(h).squeeze(-1)
        gates=model.base.decoder.tensor_gate(h).tanh()
        tr_gram=beta.double().square()+torch.einsum("...k,...kl,...l->...",gates.double(),full,gates.double())/32
        tr_source=A.double().diagonal(dim1=-2,dim2=-1).sum(-1)
        return {"h":h,"T":T,"E_pred":E,"base32":base32,"base64":base64,
            "gram":G,"gram_full":full,"V":V,"trace_gram":tr_gram,
            "trace_source":tr_source}
def load_existing_cache(part):
    assert part in ("train","val")
    with np.load(FROZEN/"runtime"/f"cache_{part}.npz",allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def verify_alignment(part,split,cache):
    assert np.array_equal(split.indices,cache["indices"])
    assert np.array_equal(split.ids,cache["ids"])
    if part=="train":
        assert np.array_equal(split.f_true,cache["f_true"])
        assert np.array_equal(split.E_true,cache["E_true"])
    assert cache["mask_f"].all() and cache["mask_E"].all()
    assert cache["h"].shape==(len(split.indices),10,128)
    assert cache["base64"].shape==(len(split.indices),10)
def compare_cached(cache,a,b,live):
    for key in ("h","E_pred","base32","base64"):
        expected=torch.as_tensor(cache[key][a:b],device=live[key].device)
        c=cfg()
        atol=c["h_cache_atol_fp32_repeat_calibrated"] if key=="h" else c["other_cache_atol"]
        if not torch.allclose(live[key],expected,rtol=1e-5,atol=atol):
            diff=float((live[key]-expected).abs().max())
            raise AssertionError(f"Frozen cache mismatch {key} max={diff}")
def gram_file_bytes(part):
    n=cfg()["train_molecules"] if part=="train" else cfg()["val_molecules"]
    return n*10*528*8

