#!/usr/bin/env python3
"""Preflight for the single frozen-head run; no training epochs."""
import copy,fcntl,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,ARCH,RUNTIME,SRC,C_F,sha,atomic_json,config,pinned_sources,Split,setup_torch,source_model,features,load_cache,head_from_initial,native_predictions,metric
def close(a,b,atol=1e-6,rtol=1e-5):
    if not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise AssertionError(f"Numerical disagreement max={float((a-b).abs().max())}")
def main():
    cfg=config();gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and gpu in (1,4,6)
    assert torch.cuda.device_count()==1
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH))
    from gpu_health import snapshot,eligible,unchanged
    before=snapshot(gpu);assert eligible(before)
    setup_torch(cfg["seed"])
    sources=pinned_sources()
    manifest=json.loads((ROOT/"CACHE_MANIFEST.json").read_text())
    assert manifest["source_hashes"]==sources and manifest["cache_script_sha256"]==sha(ROOT/"cache.py")
    assert manifest["initial_head_checkpoint_sha256"]==cfg["initial_residual_checkpoint_sha256"]
    cache={p:load_cache(p) for p in ("train","val")}
    for part in cache:
        assert sha(RUNTIME/(f"cache_{part}.npz"))==manifest["cache"][part]["sha256"]
        assert np.array_equal(cache[part]["indices"],Split(part).indices)
        assert cache[part]["mask_f"].all() and cache[part]["mask_E"].all()
    assert not np.intersect1d(cache["train"]["indices"],cache["val"]["indices"]).size
    assert len(cache["train"]["indices"])==120355 and len(cache["val"]["indices"])==6686
    model=source_model("cuda");head=model.strength_head
    assert sum(p.numel() for p in model.base.parameters())==1552092
    assert sum(p.numel() for p in head.parameters())==4161
    assert all(p.requires_grad for p in head.parameters())
    assert all(not p.requires_grad for p in model.base.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==4161
    assert not torch.count_nonzero(head[-1].weight) and not torch.count_nonzero(head[-1].bias)
    checks={}
    for part in ("train","val"):
        split=Split(part);c=cache[part];x=split.batch(0,64,"cuda")
        with torch.no_grad():
            live=features(model,x)
        for k in ("h","E_pred","base32","base64"):
            saved=torch.as_tensor(c[k][:64],device="cuda")
            close(live[k],saved,atol=1e-6,rtol=1e-5)
        with torch.no_grad():
            live_pred=native_predictions(head,live["h"],live["base32"],live["base64"],cfg["f_std_train"])
        assert torch.count_nonzero(live_pred["delta32"])==0
        assert torch.equal(live_pred["native_f64"],live["base64"].abs())
        checks[part]={"first_indices":c["indices"][:2].tolist(),"first_ids":c["ids"][:2].tolist(),
            "max_h_abs_diff":float((live["h"]-torch.as_tensor(c["h"][:64],device="cuda")).abs().max()),
            "zero_initial_delta":True,"base32_base64_distinct_paths":True}
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        assert np.array_equal(cache["val"]["ids"],z["ids"])
        assert np.array_equal(cache["val"]["indices"],z["indices"])
        assert np.array_equal(cache["val"]["f_true"],z["f_true"])
        eta0=z["f"].copy()
    ep0=metric(cache["val"]["f_true"],np.abs(cache["val"]["base64"]))["r2"]
    assert abs(ep0-0.4052941183410983)<5e-7
    # Independent copied heads: uncached frozen features and cached features.
    x=Split("train").batch(0,64,"cuda")
    with torch.no_grad():live=features(model,x)
    ca=cache["train"];cached_h=torch.as_tensor(ca["h"][:64],device="cuda")
    cached_base=torch.as_tensor(ca["base32"][:64],device="cuda")
    target=torch.as_tensor(ca["f_true"][:64].astype(np.float32),device="cuda")
    h1=copy.deepcopy(head);h2=copy.deepcopy(head)
    o1=torch.optim.Adam(h1.parameters(),lr=cfg["learning_rate"],betas=tuple(cfg["betas"]),
                         eps=cfg["eps"],weight_decay=0,amsgrad=True)
    o2=torch.optim.Adam(h2.parameters(),lr=cfg["learning_rate"],betas=tuple(cfg["betas"]),
                         eps=cfg["eps"],weight_decay=0,amsgrad=True)
    assert sum(p.numel() for g in o1.param_groups for p in g["params"])==4161
    def step(h,o,feat,base,targ):
        o.zero_grad(set_to_none=True)
        p=(base+cfg["f_std_train"]*h(feat).squeeze(-1)).abs()
        loss=(p-targ).square().mean()/cfg["train_f_variance"]
        loss.backward()
        grads=[v.grad.detach().clone() for v in h.parameters()]
        torch.nn.utils.clip_grad_norm_(h.parameters(),cfg["gradient_clip_norm"],error_if_nonfinite=True)
        o.step()
        return loss.detach(),grads
    l1,g1=step(h1,o1,live["h"],live["base32"],target)
    l2,g2=step(h2,o2,cached_h,cached_base,target)
    close(l1,l2)
    for a,b in zip(g1,g2):close(a,b)
    for a,b in zip(h1.parameters(),h2.parameters()):close(a,b)
    assert all(p.grad is None for p in model.base.parameters())
    # Two-step exact checkpoint/resume identity, including optimizer, order,
    # cursor, NumPy generator and torch RNG states.
    initial=copy.deepcopy(head.state_dict())
    def fresh():
        h=copy.deepcopy(head);h.load_state_dict(initial)
        o=torch.optim.Adam(h.parameters(),lr=cfg["learning_rate"],betas=tuple(cfg["betas"]),
            eps=cfg["eps"],weight_decay=0,amsgrad=True)
        return h,o
    positions=np.full(int(ca["indices"].max())+1,-1,dtype=np.int64)
    positions[ca["indices"]]=np.arange(len(ca["indices"]))
    rng_full=np.random.default_rng(cfg["order_seed"])
    order=rng_full.permutation(ca["indices"])
    rng_checkpoint=copy.deepcopy(rng_full.bit_generator.state)
    def batch_at(cursor):
        loc=positions[order[cursor:cursor+2]]
        assert np.all(loc>=0)
        return (torch.as_tensor(ca["h"][loc],device="cuda"),
                torch.as_tensor(ca["base32"][loc],device="cuda"),
                torch.as_tensor(ca["f_true"][loc].astype(np.float32),device="cuda"))
    first=batch_at(0);second=batch_at(2)
    h_full,o_full=fresh();h_pause,o_pause=fresh()
    step(h_full,o_full,*first);step(h_full,o_full,*second)
    step(h_pause,o_pause,*first)
    saved_h=copy.deepcopy(h_pause.state_dict());saved_o=copy.deepcopy(o_pause.state_dict())
    saved_torch=torch.get_rng_state().clone()
    saved_cuda=torch.cuda.get_rng_state().clone()
    saved_order=order.copy();saved_cursor=2
    rng_resume=np.random.default_rng();rng_resume.bit_generator.state=copy.deepcopy(rng_checkpoint)
    h_resume,o_resume=fresh();h_resume.load_state_dict(saved_h);o_resume.load_state_dict(saved_o)
    torch.set_rng_state(saved_torch);torch.cuda.set_rng_state(saved_cuda)
    assert np.array_equal(saved_order[saved_cursor:saved_cursor+2],order[2:4])
    step(h_resume,o_resume,*batch_at(saved_cursor))
    resume_difference=max(float((a-b).abs().max()) for a,b in zip(h_full.parameters(),h_resume.parameters()))
    assert resume_difference==0
    assert np.array_equal(rng_full.permutation(ca["indices"]),
                          rng_resume.permutation(ca["indices"]))
    # Preserve architecture residual global-index permutation and every epoch hash.
    residual=ARCH/"runs/retained_residual/history.jsonl"
    archived=[json.loads(x) for x in residual.read_text().splitlines()]
    rng=np.random.default_rng(cfg["order_seed"])
    hashes=[hashlib.sha256(rng.permutation(cache["train"]["indices"]).tobytes()).hexdigest()
            for _ in range(cfg["epochs"])]
    assert hashes==[r["order_sha256"] for r in archived[1:]]
    assert len(hashes)==20 and math.ceil(120355/64)==1881 and 120355%64==35
    after=snapshot(gpu);assert unchanged(before,after)
    assert set(after["apps"]).issubset({os.getpid()})
    report={"passed":True,"time":time.time(),"classification":"GPU preflight only; no training epoch",
        "physical_gpu":gpu,"gpu_before":before,"gpu_after":after,
        "source_hashes":sources,"cache_manifest_sha256":sha(ROOT/"CACHE_MANIFEST.json"),
        "cache_hashes":{p:manifest["cache"][p]["sha256"] for p in cache},
        "preflight_script_sha256":sha(Path(__file__)),"train_script_sha256":sha(ROOT/"train.py"),
        "config_sha256":sha(ROOT/"config.json"),"checks":checks,
        "initial_epoch0_native_f_r2":ep0,"cached_uncached_one_step_loss_gradient_parameter_match":True,
        "resume_max_parameter_difference":resume_difference,
        "all_20_order_hashes_match_archived_residual":True,
        "steps_per_epoch":1881,"last_batch_molecules":35,
        "only_head_trainable_parameters":4161,"no_test_loader_or_model_training":True}
    atomic_json(report,ROOT/"PREFLIGHT.json")
    print(json.dumps({"passed":True,"epoch0_r2":ep0,"resume_max_diff":resume_difference,
        "all_20_order_hashes_match":True}))
if __name__=="__main__":main()

