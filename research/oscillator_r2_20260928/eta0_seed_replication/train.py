#!/usr/bin/env python3
"""Two fresh legacy eta0 seeds; one worker per seed and exact checkpoint resume."""
import argparse,copy,fcntl,hashlib,json,math,os,random,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,ARCH,source_seal,code_hashes,config,setup,state_hash,save_json,save_torch,save_npz,sha,TrainValData,build,losses,evaluate
STOP=False
def request_stop(*_):
    global STOP
    STOP=True
def health_guard(gpu,baseline,health,phase):
    now=health.snapshot(gpu)
    if not health.unchanged(baseline,now) or not set(now["apps"]).issubset({os.getpid()}):
        save_json({"event":"INVALID_HEALTH_OR_OCCUPANCY","phase":phase,
            "baseline":baseline,"observed":now,"time":time.time()},ROOT/f"runs/seed{SEED}/INVALID.json")
        raise RuntimeError("GPU health/occupancy drift")
    return now
def rng_pack(generator):
    return {"order_numpy":copy.deepcopy(generator.bit_generator.state),
        "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state(),
        "python":random.getstate(),"numpy_global":np.random.get_state()}
def rng_restore(d,generator):
    generator.bit_generator.state=d["order_numpy"]
    torch.set_rng_state(d["torch_cpu"].cpu())
    torch.cuda.set_rng_state(d["torch_cuda"].cpu())
    random.setstate(d["python"]);np.random.set_state(d["numpy_global"])
def materialize_aliases(out,state):
    """Commit selected aliases only from versioned artifacts named by last.pt state."""
    for label,epoch_key in (("legacy","best_legacy_epoch"),("raw_f","best_raw_f_epoch")):
        epoch=state[epoch_key]
        if epoch is None:continue
        record=state["selection_artifacts"][label]
        assert record["epoch"]==epoch
        version_pt=out/"selected"/f"{label}_epoch{epoch:03d}.pt"
        version_npz=out/"selected"/f"{label}_epoch{epoch:03d}_val.npz"
        assert sha(version_pt)==record["checkpoint_sha256"]
        assert sha(version_npz)==record["predictions_sha256"]
        for src,dst in ((version_pt,out/f"best_{label}.pt"),
                        (version_npz,out/f"val_best_{label}.npz")):
            tmp=dst.with_suffix(dst.suffix+".alias_tmp")
            if tmp.exists():tmp.unlink()
            os.link(src,tmp)
            os.replace(tmp,dst)
            assert sha(dst)==sha(src)
def run(seed):
    global SEED
    SEED=seed
    c=config(seed);out=ROOT/f"runs/seed{seed}";out.mkdir(parents=True,exist_ok=True)
    workerlock=open(out/"worker.lock","a+");fcntl.flock(workerlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and gpu in (1,6)
    sys.path.insert(0,str(ARCH));import gpu_health
    glock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=gpu_health.snapshot(gpu);assert gpu_health.eligible(baseline)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"] and pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    receipt=json.loads((out/"LAUNCH_RECEIPT.json").read_text())
    assert receipt["pid"]==os.getpid() and receipt["start_ticks"]==int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21])
    assert receipt["gpu"]==gpu and receipt["gpu_uuid"]==baseline["uuid"]
    assert receipt["review_sha256"]==sha(ROOT/"IMPLEMENTATION_REVIEW.json")
    assert receipt["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    assert receipt["source_hashes"]==source and receipt["code_hashes"]==code
    assert Path.cwd()==ROOT and receipt["cwd"]==str(ROOT)
    assert not (out/"FIT_COMPLETE.json").exists() and not (out/"FAILED.json").exists() and not (out/"INVALID.json").exists()
    setup(seed)
    data=TrainValData("cuda")
    assert len(data.parts["train"])==120355 and len(data.parts["val"])==6686
    model=build(c,data.stats).cuda()
    initial_hash=state_hash(model.state_dict())
    init_pre=json.loads((ROOT/"PREFLIGHT.json").read_text())["initialization"][str(seed)]
    assert initial_hash==init_pre["state_sha256"]
    assert sum(p.numel() for p in model.parameters())==1552092
    assert sum(p.numel() for p in model.core.parameters())==1371840
    opt=torch.optim.Adam(model.parameters(),lr=c["lr"],betas=(.9,.999),eps=1e-8,
        amsgrad=True,weight_decay=c["weight_decay"])
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,
        patience=c["plateau_patience"],threshold=1e-4,threshold_mode="rel",min_lr=c["min_lr"])
    order_gen=np.random.default_rng(seed)
    plan=json.loads((ROOT/"ORDER_PLAN.json").read_text())["seeds"][str(seed)]
    state={"epoch":1,"cursor":0,"order":None,"steps":0,
        "train_sums":[0.,0.,0.],"train_counts":[0,0],
        "grad_sum":0.,"grad_max":0.,"grad_clipped":0,
        "best_legacy":None,"best_legacy_epoch":None,
        "best_raw_f_sse":None,"best_raw_f_epoch":None,
        "history":[],"total_seconds":0.,"order_hashes":[],
        "selection_artifacts":{"legacy":None,"raw_f":None}}
    initial_path=out/"initial.pt"
    if not initial_path.exists():
        save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            "epoch":0,"initial_hash":initial_hash,"seed":seed,"source_hashes":source,"code_hashes":code},initial_path)
    assert sha(initial_path)==pre["initialization"][str(seed)]["initial_pt_sha256"]
    if (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        assert ck["source_hashes"]==source and ck["code_hashes"]==code and ck["config"]==c
        assert ck["seed"]==seed and ck["initial_hash"]==initial_hash
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);sch.load_state_dict(ck["scheduler"])
        state=ck["state"];rng_restore(ck["rng"],order_gen)
        materialize_aliases(out,state)
        (out/"history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in state["history"]))
    else:
        assert not (out/"history.jsonl").exists(),"Stale history without checkpoint"
        initial_result,initial_arrays=evaluate(model,data,"val",c,with_spectrum=True)
        save_json(initial_result,out/"epoch0_val.json")
        save_npz(out/"val_epoch0.npz",**initial_arrays)
    manifest={"seed":seed,"config":c,"source_hashes":source,"code_hashes":code,
        "initial_state_sha256":initial_hash,"initial_pt_sha256":sha(initial_path),
        "parameters":{"total":1552092,"backbone":1371840},
        "normalization":data.stats,"gpu_before":baseline,
        "gpu_uuid":baseline["uuid"],"created":time.time(),
        "torch":torch.__version__,"numpy":np.__version__,
        "train_rows":120355,"val_rows":6686,"test_batches":0}
    save_json(manifest,out/"RUN_MANIFEST.json")
    tick=time.monotonic();last_ck=tick
    def checkpoint():
        nonlocal tick,last_ck
        now=time.monotonic();state["total_seconds"]+=now-tick;tick=now
        save_torch({"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),
            "rng":rng_pack(order_gen),"state":state,"seed":seed,"config":c,
            "source_hashes":source,"code_hashes":code,"initial_hash":initial_hash},out/"last.pt")
        last_ck=time.monotonic()
    checkpoint()
    while state["epoch"]<=100:
        epoch=state["epoch"];start=time.monotonic()
        if state["order"] is None:
            order=order_gen.permutation(data.parts["train"])
            order_hash=hashlib.sha256(order.tobytes()).hexdigest()
            assert order_hash==plan[epoch-1]["order_sha256"]
            assert len(order)==120355 and len(np.unique(order))==120355
            state["order"]=order;state["cursor"]=0;state["train_sums"]=[0.,0.,0.]
            state["train_counts"]=[0,0];state["grad_sum"]=0.;state["grad_max"]=0.;state["grad_clipped"]=0
            state["order_hashes"].append(order_hash)
        order=state["order"];model.train()
        while state["cursor"]<120355:
            a=state["cursor"];idx=order[a:a+64]
            x,y=data.batch(idx)
            opt.zero_grad(set_to_none=True)
            values=losses(model(**x),y,data.stats,c["eta"])
            total=values[0]
            if not torch.isfinite(total):raise FloatingPointError("nonfinite training loss")
            total.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            grad=float(gn);state["grad_sum"]+=grad;state["grad_max"]=max(state["grad_max"],grad)
            state["grad_clipped"]+=int(grad>5.)
            opt.step()
            vv=[float(v.detach()) for v in values]
            ne=int(y["mask_E"].sum());na=int(y["mask_A"].sum())
            state["train_sums"]=[s+v*n for s,v,n in zip(state["train_sums"],vv[1:],[ne,na,na])]
            state["train_counts"][0]+=ne;state["train_counts"][1]+=na
            state["cursor"]+=len(idx);state["steps"]+=1
            if state["steps"]%100==0:
                save_json({"event":"TRAIN","seed":seed,"epoch":epoch,"cursor":state["cursor"],
                    "steps":state["steps"],"lr":opt.param_groups[0]["lr"],
                    "loss":vv,"grad_norm":grad,"time":time.time()},out/"status.json")
            if state["steps"]%200==0:health_guard(gpu,baseline,gpu_health,f"step{state['steps']}")
            if STOP or time.monotonic()-last_ck>=600:
                checkpoint()
                if STOP:
                    save_json({"event":"STOPPED_CHECKPOINTED","seed":seed,
                        "epoch":epoch,"cursor":state["cursor"],"steps":state["steps"]},out/"status.json")
                    return
        # Same per-valid-element FP32 legacy objective and spectrum as original trainer.
        val,arrays=evaluate(model,data,"val",c,with_spectrum=True)
        v=val["legacy_val_objective"]
        assert all(math.isfinite(q) for q in v)
        legacy=float(v[0]);raw_sse=float(val["raw_native_f"]["sse"])
        improved_legacy=state["best_legacy"] is None or legacy<state["best_legacy"]
        improved_raw=state["best_raw_f_sse"] is None or raw_sse<state["best_raw_f_sse"]
        selected=out/"selected";selected.mkdir(exist_ok=True)
        if improved_legacy:
            state["best_legacy"]=legacy;state["best_legacy_epoch"]=epoch
            version_pt=selected/f"legacy_epoch{epoch:03d}.pt"
            version_pred=selected/f"legacy_epoch{epoch:03d}_val.npz"
            save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                "epoch":epoch,"metric":legacy,"selection":"legacy",
                "config":c,"source_hashes":source,"code_hashes":code},version_pt)
            save_npz(version_pred,**arrays)
            state["selection_artifacts"]["legacy"]={"epoch":epoch,
                "checkpoint_sha256":sha(version_pt),"predictions_sha256":sha(version_pred)}
        if improved_raw:
            state["best_raw_f_sse"]=raw_sse;state["best_raw_f_epoch"]=epoch
            version_pt=selected/f"raw_f_epoch{epoch:03d}.pt"
            version_pred=selected/f"raw_f_epoch{epoch:03d}_val.npz"
            save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                "epoch":epoch,"metric":raw_sse,"selection":"raw_f_secondary",
                "config":c,"source_hashes":source,"code_hashes":code},version_pt)
            save_npz(version_pred,**arrays)
            state["selection_artifacts"]["raw_f"]={"epoch":epoch,
                "checkpoint_sha256":sha(version_pt),"predictions_sha256":sha(version_pred)}
        prior_lr=opt.param_groups[0]["lr"]
        sch.step(legacy)
        trainloss=[state["train_sums"][0]/state["train_counts"][0]+
                   state["train_sums"][1]/state["train_counts"][1],
                   state["train_sums"][0]/state["train_counts"][0],
                   state["train_sums"][1]/state["train_counts"][1],
                   state["train_sums"][2]/state["train_counts"][1]]
        row={"epoch":epoch,"train_legacy":trainloss,
            "val_legacy":v,"val_raw_f":val["raw_native_f"],"val_energy":val["energy"],
            "val_spectrum":val["spectrum"],"val_states":val["per_state"],"val_tails":val["tails"],
            "best_legacy":state["best_legacy"],"best_legacy_epoch":state["best_legacy_epoch"],
            "best_raw_f_sse":state["best_raw_f_sse"],"best_raw_f_epoch":state["best_raw_f_epoch"],
            "lr_before_scheduler":prior_lr,"lr_after_scheduler":opt.param_groups[0]["lr"],
            "order_sha256":plan[epoch-1]["order_sha256"],
            "grad_norm_mean":state["grad_sum"]/1881,"grad_norm_max":state["grad_max"],
            "grad_clipped_updates":state["grad_clipped"],
            "updates_this_epoch":1881,"steps":state["steps"],"seconds":time.monotonic()-start,
            "peak_memory_bytes":torch.cuda.max_memory_allocated(),"time":time.time()}
        state["history"].append(row)
        state["epoch"]+=1;state["order"]=None;state["cursor"]=0
        checkpoint()
        materialize_aliases(out,state)
        (out/"history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in state["history"]))
        save_json({"event":"EPOCH_COMPLETE","seed":seed,**row},out/"status.json")
        health_guard(gpu,baseline,gpu_health,f"epoch{epoch}")
        if STOP:return
    assert state["steps"]==188100 and len(state["history"])==100
    assert state["order_hashes"]==[r["order_sha256"] for r in plan]
    save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        "epoch":100,"selection":"final","config":c,"source_hashes":source,"code_hashes":code},out/"final100.pt")
    final_val,final_arrays=evaluate(model,data,"val",c,with_spectrum=False)
    save_npz(out/"val_final100.npz",**final_arrays)
    # Full training metrics at fixed checkpoints; these cannot select a checkpoint.
    fulltrain={}
    for label,path in (("initial",initial_path),("legacy",out/"best_legacy.pt"),
                       ("raw_f",out/"best_raw_f.pt"),("final100",out/"final100.pt")):
        ck=torch.load(path,map_location="cuda",weights_only=False)
        model.load_state_dict(ck["model"])
        val_fixed,val_arrays=evaluate(model,data,"val",c,with_spectrum=False)
        saved_name={"initial":"epoch0","legacy":"best_legacy",
                    "raw_f":"best_raw_f","final100":"final100"}[label]
        with np.load(out/f"val_{saved_name}.npz",allow_pickle=False) as z:
            assert np.array_equal(z["ids"],val_arrays["ids"])
            assert np.array_equal(z["indices"],val_arrays["indices"])
            assert np.array_equal(z["f_true"],val_arrays["f_true"])
            assert np.array_equal(z["E_true"],val_arrays["E_true"])
            replay_f_max=float(np.abs(z["f"]-val_arrays["f"]).max())
            replay_E_max=float(np.abs(z["E"]-val_arrays["E"]).max())
            assert np.allclose(z["f"],val_arrays["f"],rtol=1e-4,atol=2e-5)
            assert np.allclose(z["E"],val_arrays["E"],rtol=1e-5,atol=1e-5)
        train_fixed,_=evaluate(model,data,"train",c,with_spectrum=False)
        fulltrain[label]={"epoch":ck["epoch"],"checkpoint_sha256":sha(path),
            "val_prediction_replay_max_abs_f":replay_f_max,
            "val_prediction_replay_max_abs_E":replay_E_max,
            "replay_tolerances":{"f_atol":2e-5,"f_rtol":1e-4,"E_atol":1e-5,"E_rtol":1e-5},
            "train":train_fixed,"val":val_fixed}
        save_json(fulltrain,out/"FIXED_CHECKPOINT_METRICS.json")
        health_guard(gpu,baseline,gpu_health,f"fixed_{label}")
    assert fulltrain["legacy"]["epoch"]==state["best_legacy_epoch"]
    assert fulltrain["raw_f"]["epoch"]==state["best_raw_f_epoch"]
    with np.load(out/"val_best_legacy.npz",allow_pickle=False) as z:
        saved_legacy_f=z["f"].copy();saved_truth=z["f_true"].copy()
    with np.load(out/"val_best_raw_f.npz",allow_pickle=False) as z:
        saved_raw_f=z["f"].copy();assert np.array_equal(saved_truth,z["f_true"])
    assert abs(float(np.square(saved_raw_f-saved_truth).sum())-state["best_raw_f_sse"])<1e-8
    assert np.isfinite(fulltrain["legacy"]["val"]["legacy_val_objective"][0])
    assert abs(fulltrain["legacy"]["val"]["legacy_val_objective"][0]-state["best_legacy"])<=1e-5
    assert abs(fulltrain["raw_f"]["val"]["raw_native_f"]["sse"]-state["best_raw_f_sse"])<=1e-4
    receipt={"event":"FIT_COMPLETE","seed":seed,"epochs":100,"steps":state["steps"],
        "best_legacy_epoch":state["best_legacy_epoch"],"best_raw_f_epoch":state["best_raw_f_epoch"],
        "best_legacy":state["best_legacy"],"best_raw_f_sse":state["best_raw_f_sse"],
        "source_hashes":source,"code_hashes":code,
        "initial_hash":initial_hash,"order_hashes_sha256":sha(ROOT/"ORDER_PLAN.json"),
        "checkpoint_hashes":{label:sha(path) for label,path in (
            ("initial",initial_path),("legacy",out/"best_legacy.pt"),
            ("raw_f",out/"best_raw_f.pt"),("final100",out/"final100.pt"),("last",out/"last.pt"))},
        "val_prediction_hashes":{label:sha(out/f"val_{label}.npz") for label in
            ("epoch0","best_legacy","best_raw_f","final100")},
        "fixed_metrics_sha256":sha(out/"FIXED_CHECKPOINT_METRICS.json"),
        "gpu_before":baseline,"gpu_after":health_guard(gpu,baseline,gpu_health,"complete"),
        "seconds":state["total_seconds"],"time":time.time(),"test_batches":0}
    save_json(receipt,out/"FIT_COMPLETE.json")
    save_json({"event":"FIT_COMPLETE","seed":seed,"time":time.time()},out/"status.json")
    print(json.dumps({"seed":seed,"complete":True,"best_legacy_epoch":state["best_legacy_epoch"],
        "best_raw_f_epoch":state["best_raw_f_epoch"]}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("seed",type=int,choices=[23,37]);a=ap.parse_args()
    try:run(a.seed)
    except BaseException as exc:
        out=ROOT/f"runs/seed{a.seed}";out.mkdir(parents=True,exist_ok=True)
        save_json({"event":"FAILED","type":type(exc).__name__,"error":str(exc),
            "traceback":traceback.format_exc(),"time":time.time()},out/"FAILED.json")
        raise

