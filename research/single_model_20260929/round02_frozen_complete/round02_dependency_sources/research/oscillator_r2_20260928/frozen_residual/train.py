#!/usr/bin/env python3
"""Single frozen eta0 residual-head run; head-only optimization on audited caches."""
import copy,fcntl,hashlib,json,os,random,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,ARCH,RUNTIME,SRC,sha,atomic_json,atomic_npz,config,pinned_sources,load_cache,head_from_initial,native_predictions,metric,metrics,cache_arrays_gpu,setup_torch
STOP=False
def request_stop(*_):
    global STOP
    STOP=True
def atomic_torch(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def current_hashes():
    paths=[ROOT/n for n in ("FROZEN_RESIDUAL_PROTOCOL.md","config.json","common.py",
        "cache.py","preflight.py","train.py","summarize.py","launch.py",
        "CACHE_MANIFEST.json","PREFLIGHT.json")]
    paths += [ARCH/"INDEPENDENT_COMPLETION_AUDIT.json",
        ROOT.parent/"completion_receipts/FINAL20_REPLAY_RESULTS.json"]
    return {str(p):sha(p) for p in paths}
def evaluate(head,cache_np,cache_gpu,cfg,part):
    head.eval()
    pred_f=[];signed=[];delta=[]
    with torch.inference_mode():
        for a in range(0,len(cache_np["ids"]),cfg["eval_batch_size_molecules"]):
            b=min(a+cfg["eval_batch_size_molecules"],len(cache_np["ids"]))
            q=native_predictions(head,cache_gpu["h"][a:b],cache_gpu["base32"][a:b],
                cache_gpu["base64"][a:b],cfg["f_std_train"])
            pred_f.append(q["native_f64"].cpu().numpy())
            if part=="val":
                signed.append(q["signed64"].cpu().numpy())
                delta.append(q["delta32"].double().cpu().numpy())
    f=np.concatenate(pred_f)
    out={"f":f}
    if part=="val":
        out["signed"]=np.concatenate(signed);out["delta"]=np.concatenate(delta)
    report=metrics(cache_np,out,cfg["f_std_train"],cfg["train_f_variance"],
                   cfg["train_E_variance"],
                   cfg["raw_validation_q90"] if part=="val" else None,
                   cfg["raw_validation_q99"] if part=="val" else None)
    if part=="val":
        r=report["residual"]
        for name,array in (("base",cache_np["base64"]),("signed",out["signed"]),
                           ("delta",out["delta"])):
            r[name+"_quantiles"]=[float(v) for v in np.quantile(array,[0,.01,.5,.99,1])]
            r[name+"_min"]=float(array.min())
            r[name+"_max"]=float(array.max())
    return report,out
def val_arrays(cache,pred):
    return {"ids":cache["ids"],"indices":cache["indices"],"f_true":cache["f_true"],
        "E_true":cache["E_true"],"E_pred":cache["E_pred"],
        "base_f32":cache["base32"],"base_f_native64":cache["base64"],
        "delta_f":pred["delta"],"signed_f":pred["signed"],"f_pred":pred["f"]}
def main():
    cfg=config();gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu)
    assert gpu in cfg["allowed_physical_gpus"] and torch.cuda.device_count()==1
    out=ROOT/"run";out.mkdir(exist_ok=True)
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ownlock=open(out/"worker.lock","a+");fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH))
    from gpu_health import snapshot,eligible,unchanged
    before=snapshot(gpu);assert eligible(before,guarded=gpu==5)
    if gpu==5:raise RuntimeError("Guarded GPU5 launch requires separate numerical microcheck")
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    setup_torch(cfg["seed"])
    source=pinned_sources();manifest=json.loads((ROOT/"CACHE_MANIFEST.json").read_text())
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert manifest["source_hashes"]==source
    assert pre["passed"] and pre["source_hashes"]==source
    assert pre["cache_manifest_sha256"]==sha(ROOT/"CACHE_MANIFEST.json")
    assert pre["train_script_sha256"]==sha(ROOT/"train.py")
    hashes=current_hashes()
    assert review["passed"] and review["source_hashes"]==hashes
    assert not (out/"FIT_COMPLETE.json").exists()
    if (out/"run_manifest.json").exists() and not (out/"last.pt").exists():
        raise RuntimeError("Stale output without resumable checkpoint")
    cache={k:load_cache(k) for k in ("train","val")}
    for part in cache:
        assert sha(RUNTIME/(f"cache_{part}.npz"))==manifest["cache"][part]["sha256"]
        assert cache[part]["mask_f"].all() and cache[part]["mask_E"].all()
    train=cache["train"];val=cache["val"]
    assert train["h"].shape==(120355,10,128) and val["h"].shape==(6686,10,128)
    tc={k:cache_arrays_gpu(v,"cuda") for k,v in cache.items()}
    head=head_from_initial("cuda")
    optimizer=torch.optim.Adam(head.parameters(),lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),eps=cfg["eps"],weight_decay=cfg["weight_decay"],
        amsgrad=cfg["amsgrad"])
    assert sum(p.numel() for g in optimizer.param_groups for p in g["params"])==4161
    generator=np.random.default_rng(cfg["order_seed"])
    positions=np.full(int(train["indices"].max())+1,-1,dtype=np.int64)
    positions[train["indices"]]=np.arange(len(train["indices"]))
    residual_hist=[json.loads(s) for s in (ARCH/"runs/retained_residual/history.jsonl").read_text().splitlines()]
    expected_orders=[x["order_sha256"] for x in residual_hist[1:]]
    state={"epoch":1,"cursor":0,"order":None,"steps":0,"history":[],
        "best_sse":None,"best_epoch":None,"online_sum":0.,"online_count":0,
        "grad_norm_sum":0.,"grad_norm_max":0.,"clipped_steps":0,
        "cache_seconds":{k:manifest["cache"][k]["generation_seconds"] for k in cache},
        "training_seconds":0.,"evaluation_seconds":0.}
    if (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        assert ck["source_hashes"]==hashes and ck["config"]==cfg
        head.load_state_dict(ck["head"]);optimizer.load_state_dict(ck["optimizer"]);state=ck["state"]
        generator.bit_generator.state=ck["rng_numpy"]
        torch.set_rng_state(ck["rng_torch"].cpu())
        torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        random.setstate(ck["rng_python"]);np.random.set_state(ck["rng_numpy_global"])
        with open(out/"history.jsonl","w") as f:
            for row in state["history"]:f.write(json.dumps(row,allow_nan=False)+"\n")
    initial_head_hash=sha(ARCH/"initial/retained_residual.pt")
    atomic_json({"classification":"Head-only frozen eta0 residual run","config":cfg,
        "source_hashes":hashes,"cache_hashes":{k:manifest["cache"][k]["sha256"] for k in cache},
        "physical_gpu":gpu,"gpu_uuid":before["uuid"],"gpu_before":before,
        "initial_head_checkpoint_sha256":initial_head_hash,"pid":os.getpid(),
        "review_sha256":sha(ROOT/"IMPLEMENTATION_REVIEW.json"),
        "preflight_sha256":sha(ROOT/"PREFLIGHT.json"),"time":time.time()},out/"run_manifest.json")
    def guard(epoch):
        now=snapshot(gpu)
        if not unchanged(before,now) or not set(now["apps"]).issubset({os.getpid()}):
            atomic_json({"event":"INVALID_GPU_HEALTH_OR_OCCUPANCY","epoch":epoch,
                "before":before,"observed":now,"time":time.time()},out/"INVALID.json")
            raise RuntimeError("GPU health or ownership changed")
        return now
    def checkpoint():
        atomic_torch({"config":cfg,"source_hashes":hashes,"head":head.state_dict(),
            "optimizer":optimizer.state_dict(),"state":state,
            "rng_numpy":generator.bit_generator.state,"rng_torch":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state(),"rng_python":random.getstate(),
            "rng_numpy_global":np.random.get_state()},out/"last.pt")
    def commit():
        checkpoint()
        with open(out/"history.jsonl","w") as f:
            for row in state["history"]:f.write(json.dumps(row,allow_nan=False)+"\n")
        if state["best_epoch"] is not None:
            sel=out/"selections"/f"epoch{state['best_epoch']:03d}.pt"
            vals=out/"selections"/f"epoch{state['best_epoch']:03d}_val.npz"
            assert sel.exists() and vals.exists()
            for src,name in ((sel,"best_native_f.pt"),(vals,"best_native_f_val.npz")):
                tmp=out/(name+".tmp")
                if tmp.exists():tmp.unlink()
                os.link(src,tmp);os.replace(tmp,out/name)
    def record(epoch,order_hash=None,online_mean=None,grad=None):
        h0=guard(epoch);start=time.monotonic()
        train_metrics,_=evaluate(head,train,tc["train"],cfg,"train")
        val_metrics,pred=evaluate(head,val,tc["val"],cfg,"val")
        torch.cuda.synchronize()
        state["evaluation_seconds"]+=time.monotonic()-start
        h1=guard(epoch)
        sse=val_metrics["raw_f"]["sse"]
        improved=state["best_sse"] is None or sse<state["best_sse"]
        if improved:
            folder=out/"selections";folder.mkdir(exist_ok=True)
            atomic_torch({"epoch":epoch,"head":head.state_dict(),"validation":val_metrics,
                "train":train_metrics,"source_hashes":hashes},folder/f"epoch{epoch:03d}.pt")
            atomic_npz(val_arrays(val,pred),folder/f"epoch{epoch:03d}_val.npz")
            state["best_sse"]=sse;state["best_epoch"]=epoch
        if epoch==0:
            assert abs(val_metrics["raw_f"]["r2"]-0.4052941183410983)<5e-7
            atomic_npz(val_arrays(val,pred),out/"initial_val.npz")
        if epoch==20:atomic_npz(val_arrays(val,pred),out/"final20_val.npz")
        row={"epoch":epoch,"validation":val_metrics,"train":train_metrics,
            "selected_best_epoch_so_far":state["best_epoch"],
            "selected_best_sse_so_far":state["best_sse"],
            "order_sha256":order_hash,"online_training_objective":online_mean,
            "head_gradients":copy.deepcopy(grad),"steps":state["steps"],
            "gpu_health_before":h0,"gpu_health_after":h1,"time":time.time()}
        state["history"].append(row)
        atomic_json(row,out/"status.json")
        print(json.dumps({"epoch":epoch,"val_f_r2":val_metrics["raw_f"]["r2"],
            "train_f_r2":train_metrics["raw_f"]["r2"],
            "best_epoch":state["best_epoch"],"steps":state["steps"]}),flush=True)
    if not state["history"]:
        record(0);commit()
    while state["epoch"]<=cfg["epochs"]:
        epoch=state["epoch"];start=time.monotonic();guard(epoch)
        if state["order"] is None:
            state["order"]=generator.permutation(train["indices"])
            state["cursor"]=0;state["online_sum"]=0.;state["online_count"]=0
            state["grad_norm_sum"]=0.;state["grad_norm_max"]=0.;state["clipped_steps"]=0
        order=state["order"]
        order_hash=hashlib.sha256(order.tobytes()).hexdigest()
        assert order_hash==expected_orders[epoch-1]
        head.train()
        while state["cursor"]<len(order):
            ix=order[state["cursor"]:state["cursor"]+cfg["batch_size_molecules"]]
            loc=positions[ix]
            assert np.all(loc>=0)
            optimizer.zero_grad(set_to_none=True)
            pos=torch.as_tensor(loc,device="cuda")
            h=tc["train"]["h"][pos];base=tc["train"]["base32"][pos];target=tc["train"]["f32"][pos]
            pred=(base+cfg["f_std_train"]*head(h).squeeze(-1)).abs()
            loss=(pred-target).square().mean()/cfg["train_f_variance"]
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),cfg["gradient_clip_norm"],
                error_if_nonfinite=True)
            optimizer.step()
            normf=float(norm);state["grad_norm_sum"]+=normf
            state["grad_norm_max"]=max(state["grad_norm_max"],normf)
            state["clipped_steps"]+=int(normf>cfg["gradient_clip_norm"])
            state["online_sum"]+=float(loss.detach())*len(loc)
            state["online_count"]+=len(loc)
            state["cursor"]+=len(loc);state["steps"]+=1
            if state["steps"]%100==0:
                guard(epoch);checkpoint()
                atomic_json({"event":"TRAIN","epoch":epoch,"cursor":state["cursor"],
                    "steps":state["steps"],"f_loss":float(loss.detach()),
                    "grad_norm":normf,"time":time.time()},out/"status.json")
            if STOP:
                checkpoint()
                atomic_json({"event":"STOPPED_CHECKPOINTED","epoch":epoch,
                    "cursor":state["cursor"],"steps":state["steps"],
                    "time":time.time()},out/"status.json")
                return
        torch.cuda.synchronize()
        state["training_seconds"]+=time.monotonic()-start
        steps_this=cfg["expected_steps_per_epoch"]
        grad={"mean_unclipped_norm":state["grad_norm_sum"]/steps_this,
            "max_unclipped_norm":state["grad_norm_max"],
            "clipped_steps":state["clipped_steps"],"steps":steps_this}
        record(epoch,order_hash,state["online_sum"]/state["online_count"],grad)
        state["epoch"]+=1;state["cursor"]=0;state["order"]=None
        commit()
        if STOP:return
    assert state["steps"]==cfg["expected_total_steps"]
    assert state["best_epoch"]==min(range(21),key=lambda j:state["history"][j]["validation"]["raw_f"]["sse"])
    assert sha(RUNTIME/"cache_train.npz")==manifest["cache"]["train"]["sha256"]
    assert sha(RUNTIME/"cache_val.npz")==manifest["cache"]["val"]["sha256"]
    after=guard(20)
    atomic_json({"event":"FIT_COMPLETE","epochs":20,"steps":state["steps"],
        "best_epoch":state["best_epoch"],"best_sse":state["best_sse"],
        "best_raw_f_r2":state["history"][state["best_epoch"]]["validation"]["raw_f"]["r2"],
        "final20_raw_f_r2":state["history"][-1]["validation"]["raw_f"]["r2"],
        "head_parameters":4161,"source_hashes":hashes,
        "cache_hashes":{k:manifest["cache"][k]["sha256"] for k in cache},
        "gpu_before":before,"gpu_after":after,
        "cache_seconds":state["cache_seconds"],"training_seconds":state["training_seconds"],
        "evaluation_seconds":state["evaluation_seconds"],"time":time.time()},out/"FIT_COMPLETE.json")
    print("FIT_COMPLETE frozen_residual",flush=True)
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        out=ROOT/"run";out.mkdir(exist_ok=True)
        atomic_json({"type":type(exc).__name__,"error":str(exc),"time":time.time()},
            out/"FAILED.json")
        raise

