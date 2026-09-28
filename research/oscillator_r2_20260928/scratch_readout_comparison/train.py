#!/usr/bin/env python3
"""Reviewed paired scratch worker. One fixed arm, exact resume and committed selection."""
import argparse,copy,fcntl,hashlib,json,math,os,random,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,source_seal,code_hashes,cfg,stats,setup,sha,save_json,save_torch,save_npz,objective,evaluate,lr_for_epoch,TrainValData,counts
from model import NativeDirect,MTODirect
STOP=False
def request_stop(*_):
    global STOP
    STOP=True
def rng_pack(g):
    return {"order_numpy":copy.deepcopy(g.bit_generator.state),"torch_cpu":torch.get_rng_state(),
        "torch_cuda":torch.cuda.get_rng_state(),"python":random.getstate(),"numpy_global":np.random.get_state()}
def rng_restore(d,g):
    g.bit_generator.state=d["order_numpy"];torch.set_rng_state(d["torch_cpu"].cpu())
    torch.cuda.set_rng_state(d["torch_cuda"].cpu());random.setstate(d["python"]);np.random.set_state(d["numpy_global"])
def health_guard(gpu,baseline,health,arm,phase):
    current=health.snapshot(gpu)
    if not health.unchanged(baseline,current) or not set(current["apps"]).issubset({os.getpid()}):
        save_json({"event":"INVALID_HEALTH_OR_OCCUPANCY","phase":phase,"baseline":baseline,
            "observed":current,"time":time.time()},ROOT/f"runs/{arm}/INVALID.json")
        raise RuntimeError("GPU health/occupancy changed")
    return current
def aliases(out,state):
    for label in ("raw_f","joint"):
        rec=state["selection_artifacts"][label]
        assert rec and rec["epoch"]==state["best"][label]["epoch"]
        ep=rec["epoch"];s=out/"selected"/f"{label}_epoch{ep:03d}"
        for src,dst,digest in ((Path(str(s)+".pt"),out/f"best_{label}.pt",rec["checkpoint_sha256"]),
                (Path(str(s)+"_val.npz"),out/f"val_best_{label}.npz",rec["predictions_sha256"])):
            assert sha(src)==digest
            tmp=dst.with_suffix(dst.suffix+".alias_tmp")
            if tmp.exists():tmp.unlink()
            os.link(src,tmp);os.replace(tmp,dst)
            assert sha(dst)==digest
def train(arm):
    c=cfg();st=stats();out=ROOT/"runs"/arm;out.mkdir(parents=True,exist_ok=True)
    flock=open(out/"worker.lock","a+");fcntl.flock(flock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and gpu in (2,4,5)
    arch=ROOT.parent/"architecture";sys.path.insert(0,str(arch));import gpu_health
    glock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=gpu_health.snapshot(gpu);assert gpu_health.eligible(baseline,guarded=(gpu==5))
    if gpu==5:
        admit=json.loads((ROOT/"GPU5_ADMISSION.json").read_text())
        assert admit["passed"] and admit["gpu_uuid"]==baseline["uuid"]
        assert admit["snapshot"]["ecc"]==baseline["ecc"] and admit["microcheck_passed"]
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    source=source_seal();code=code_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text());review=json.loads((ROOT/"IMPLEMENTATION_REVIEW.json").read_text())
    assert pre["passed"] and review["passed"]
    assert pre["source_hashes"]==review["source_hashes"]==source
    assert pre["code_hashes"]==review["code_hashes"]==code
    assert review["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    receipt_path=out/"LAUNCH_RECEIPT.json"
    deadline=time.monotonic()+30
    while not receipt_path.exists() and time.monotonic()<deadline:time.sleep(.1)
    assert receipt_path.exists(),"Launch receipt missing"
    receipt=json.loads(receipt_path.read_text())
    assert receipt["pid"]==os.getpid() and receipt["start_ticks"]==int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21])
    assert receipt["gpu"]==gpu and receipt["gpu_uuid"]==baseline["uuid"]
    assert receipt["review_sha256"]==sha(ROOT/"IMPLEMENTATION_REVIEW.json")
    assert receipt["preflight_sha256"]==sha(ROOT/"PREFLIGHT.json")
    assert receipt["source_hashes"]==source and receipt["code_hashes"]==code
    if gpu==5:
        assert receipt["gpu5_admission_sha256"]==sha(ROOT/"GPU5_ADMISSION.json")
        assert receipt["gpu5_admission_review_sha256"]==sha(ROOT/"GPU5_ADMISSION_REVIEW.json")
        review5=json.loads((ROOT/"GPU5_ADMISSION_REVIEW.json").read_text())
        assert review5["passed"] and review5["admission_sha256"]==receipt["gpu5_admission_sha256"]
    assert Path.cwd()==ROOT and receipt["cwd"]==str(ROOT)
    assert not any((out/n).exists() for n in ("FIT_COMPLETE.json","FAILED.json","INVALID.json"))
    setup(11);data=TrainValData("cuda")
    init=torch.load(ROOT/"initial"/f"{arm}.pt",map_location="cpu",weights_only=False)
    model=(NativeDirect(st) if arm=="native" else MTODirect(st)).cuda()
    model.load_state_dict(init["model"]);initial_hash=init["state_sha256"]
    assert sum(p.numel() for p in model.parameters())==counts(model)["total"]
    opt=torch.optim.Adam(model.parameters(),lr=lr_for_epoch(1),betas=(.9,.999),eps=1e-8,
        amsgrad=True,weight_decay=0)
    g=np.random.default_rng(11);plan=json.loads((ROOT/"ORDER_PLAN.json").read_text())["orders"]
    state={"epoch":1,"cursor":0,"order":None,"steps":0,"history":[],"order_hashes":[],
        "sums":[0.,0.],"counts":[0,0],"grad_sum":0.,"grad_max":0.,"grad_clip":0,
        "best":{"raw_f":None,"joint":None},"selection_artifacts":{"raw_f":None,"joint":None},
        "seconds":0.}
    if (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        assert ck["arm"]==arm and ck["source_hashes"]==source and ck["code_hashes"]==code and ck["config"]==c
        assert ck["initial_hash"]==initial_hash
        model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
        state=ck["state"];rng_restore(ck["rng"],g);aliases(out,state)
        (out/"history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in state["history"]),encoding="utf-8")
    else:
        assert not (out/"history.jsonl").exists()
        result,arrays=evaluate(model,data,"val",st,c)
        save_json(result,out/"epoch0_val.json");save_npz(out/"val_epoch0.npz",**arrays)
        selected=out/"selected";selected.mkdir(exist_ok=True)
        for label,val in (("raw_f",result["raw_f"]["sse"]),("joint",result["normalized_joint"][0])):
            prefix=selected/f"{label}_epoch000"
            save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                "epoch":0,"metric":val,"selection":label,"arm":arm,
                "source_hashes":source,"code_hashes":code},Path(str(prefix)+".pt"))
            save_npz(Path(str(prefix)+"_val.npz"),**arrays)
            state["best"][label]={"epoch":0,"metric":val}
            state["selection_artifacts"][label]={"epoch":0,
                "checkpoint_sha256":sha(Path(str(prefix)+".pt")),
                "predictions_sha256":sha(Path(str(prefix)+"_val.npz"))}
    save_json({"arm":arm,"config":c,"stats":st,"initial_hash":initial_hash,
        "source_hashes":source,"code_hashes":code,"gpu_before":baseline,
        "gpu_uuid":baseline["uuid"],"counts":counts(model),"torch":torch.__version__,
        "numpy":np.__version__,"time":time.time(),"test_batches":0},out/"RUN_MANIFEST.json")
    tick=time.monotonic();last_ck=tick
    def checkpoint():
        nonlocal tick,last_ck
        now=time.monotonic();state["seconds"]+=now-tick;tick=now
        save_torch({"model":model.state_dict(),"optimizer":opt.state_dict(),
            "rng":rng_pack(g),"state":state,"arm":arm,"config":c,
            "source_hashes":source,"code_hashes":code,"initial_hash":initial_hash},out/"last.pt")
        last_ck=time.monotonic()
    checkpoint();aliases(out,state)
    while state["epoch"]<=100:
        ep=state["epoch"];start=time.monotonic()
        lr=lr_for_epoch(ep)
        for group in opt.param_groups:group["lr"]=lr
        if state["order"] is None:
            order=g.permutation(data.parts["train"])
            digest=hashlib.sha256(order.tobytes()).hexdigest()
            assert digest==plan[ep-1]["sha256"]
            assert len(order)==120355 and len(np.unique(order))==120355
            state["order"]=order;state["cursor"]=0;state["sums"]=[0.,0.]
            state["counts"]=[0,0];state["grad_sum"]=0.;state["grad_max"]=0.;state["grad_clip"]=0
            state["order_hashes"].append(digest)
        order=state["order"];model.train()
        while state["cursor"]<120355:
            ix=order[state["cursor"]:state["cursor"]+64]
            x,y=data.batch(ix)
            opt.zero_grad(set_to_none=True)
            total,le,lf=objective(model(**x),y,st)
            if not torch.isfinite(total):raise FloatingPointError("nonfinite training loss")
            total.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            grad=float(gn);state["grad_sum"]+=grad;state["grad_max"]=max(state["grad_max"],grad)
            state["grad_clip"]+=int(grad>5.)
            opt.step()
            ne=int(y["mask_E"].sum());nf=int(y["mask_f"].sum())
            state["sums"][0]+=float(le.detach())*ne;state["sums"][1]+=float(lf.detach())*nf
            state["counts"][0]+=ne;state["counts"][1]+=nf
            state["cursor"]+=len(ix);state["steps"]+=1
            if state["steps"]%100==0:
                save_json({"event":"TRAIN","arm":arm,"epoch":ep,"cursor":state["cursor"],
                    "steps":state["steps"],"lr":lr,"loss":[float(total),float(le),float(lf)],
                    "grad_norm":grad,"time":time.time()},out/"status.json")
            if state["steps"]%200==0:health_guard(gpu,baseline,gpu_health,arm,f"step{state['steps']}")
            if STOP or time.monotonic()-last_ck>=600:
                checkpoint()
                if STOP:
                    save_json({"event":"STOPPED_CHECKPOINTED","arm":arm,"epoch":ep,
                        "cursor":state["cursor"],"steps":state["steps"]},out/"status.json")
                    return
        val,arrays=evaluate(model,data,"val",st,c)
        selected=out/"selected"
        for label,val_metric in (("raw_f",val["raw_f"]["sse"]),("joint",val["normalized_joint"][0])):
            assert math.isfinite(val_metric)
            if val_metric<state["best"][label]["metric"]:
                prefix=selected/f"{label}_epoch{ep:03d}"
                save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "epoch":ep,"metric":val_metric,"selection":label,"arm":arm,
                    "source_hashes":source,"code_hashes":code},Path(str(prefix)+".pt"))
                save_npz(Path(str(prefix)+"_val.npz"),**arrays)
                state["best"][label]={"epoch":ep,"metric":val_metric}
                state["selection_artifacts"][label]={"epoch":ep,
                    "checkpoint_sha256":sha(Path(str(prefix)+".pt")),
                    "predictions_sha256":sha(Path(str(prefix)+"_val.npz"))}
        row={"epoch":ep,"train_normalized":[sum(state["sums"][j]/state["counts"][j] for j in range(2)),
            *[state["sums"][j]/state["counts"][j] for j in range(2)]],
            "val_normalized":val["normalized_joint"],"val_raw_f":val["raw_f"],
            "val_energy":val["energy"],"best":copy.deepcopy(state["best"]),
            "order_sha256":plan[ep-1]["sha256"],"lr":lr,
            "grad_norm_mean":state["grad_sum"]/1881,"grad_norm_max":state["grad_max"],
            "grad_clipped_updates":state["grad_clip"],"updates_this_epoch":1881,
            "steps":state["steps"],"seconds":time.monotonic()-start,
            "peak_memory_bytes":torch.cuda.max_memory_allocated(),"time":time.time()}
        state["history"].append(row);state["epoch"]+=1;state["cursor"]=0;state["order"]=None
        checkpoint();aliases(out,state)
        (out/"history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in state["history"]),encoding="utf-8")
        save_json({"event":"EPOCH_COMPLETE","arm":arm,**row},out/"status.json")
        health_guard(gpu,baseline,gpu_health,arm,f"epoch{ep}")
        if STOP:return
    assert state["steps"]==188100 and len(state["history"])==100
    assert state["order_hashes"]==[r["sha256"] for r in plan]
    save_torch({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        "epoch":100,"selection":"final","arm":arm,"source_hashes":source,
        "code_hashes":code},out/"final100.pt")
    final_val,final_arrays=evaluate(model,data,"val",st,c)
    save_npz(out/"val_final100.npz",**final_arrays)
    full={}
    for label,path,arrname in (("initial",ROOT/"initial"/f"{arm}.pt","epoch0"),
        ("raw_f",out/"best_raw_f.pt","best_raw_f"),
        ("joint",out/"best_joint.pt","best_joint"),
        ("final100",out/"final100.pt","final100")):
        ck=torch.load(path,map_location="cuda",weights_only=False)
        model.load_state_dict(ck["model"]);v,va=evaluate(model,data,"val",st,c)
        with np.load(out/f"val_{arrname}.npz",allow_pickle=False) as z:
            assert np.array_equal(z["ids"],va["ids"]) and np.array_equal(z["indices"],va["indices"])
            assert np.array_equal(z["f_true"],va["f_true"]) and np.array_equal(z["E_true"],va["E_true"])
            fmax=float(np.abs(z["f"]-va["f"]).max());emax=float(np.abs(z["E"]-va["E"]).max())
            assert np.allclose(z["f"],va["f"],rtol=1e-4,atol=2e-5)
            assert np.allclose(z["E"],va["E"],rtol=1e-5,atol=1e-5)
        t,_=evaluate(model,data,"train",st,c)
        full[label]={"epoch":ck.get("epoch",0),"checkpoint_sha256":sha(path),
            "val_replay_max_abs_f":fmax,"val_replay_max_abs_E":emax,"train":t,"val":v}
        save_json(full,out/"FIXED_CHECKPOINT_METRICS.json")
        health_guard(gpu,baseline,gpu_health,arm,f"fixed_{label}")
    assert full["raw_f"]["epoch"]==state["best"]["raw_f"]["epoch"]
    assert full["joint"]["epoch"]==state["best"]["joint"]["epoch"]
    with np.load(out/"val_best_raw_f.npz",allow_pickle=False) as z:
        assert abs(float(np.square(z["f"]-z["f_true"]).sum())-
            state["best"]["raw_f"]["metric"])<1e-8
    receipt={"event":"FIT_COMPLETE","arm":arm,"epochs":100,"steps":state["steps"],
        "best":state["best"],"source_hashes":source,"code_hashes":code,
        "initial_hash":initial_hash,"order_plan_sha256":sha(ROOT/"ORDER_PLAN.json"),
        "checkpoint_hashes":{label:sha(path) for label,path in (
            ("initial",ROOT/"initial"/f"{arm}.pt"),("raw_f",out/"best_raw_f.pt"),
            ("joint",out/"best_joint.pt"),("final100",out/"final100.pt"),("last",out/"last.pt"))},
        "prediction_hashes":{label:sha(out/f"val_{label}.npz") for label in
            ("epoch0","best_raw_f","best_joint","final100")},
        "fixed_metrics_sha256":sha(out/"FIXED_CHECKPOINT_METRICS.json"),
        "gpu_before":baseline,"gpu_after":health_guard(gpu,baseline,gpu_health,arm,"complete"),
        "seconds":state["seconds"],"time":time.time(),"test_batches":0}
    save_json(receipt,out/"FIT_COMPLETE.json")
    save_json({"event":"FIT_COMPLETE","arm":arm,"time":time.time()},out/"status.json")
    print(json.dumps({"arm":arm,"complete":True,"best":state["best"]}),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("arm",choices=["native","mto"]);a=ap.parse_args()
    try:train(a.arm)
    except BaseException as exc:
        out=ROOT/"runs"/a.arm;out.mkdir(parents=True,exist_ok=True)
        save_json({"event":"FAILED","type":type(exc).__name__,"error":str(exc),
            "traceback":traceback.format_exc(),"time":time.time()},out/"FAILED.json")
        raise

