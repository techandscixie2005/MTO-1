#!/usr/bin/env python3
"""Three-arm, validation-selected continuation from frozen eta0 weights."""
import argparse, copy, fcntl, hashlib, json, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(SRC))
from dataset import Data
from model_factory import build
from pilot_objective import terms, C_F

STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(value,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(tmp,p)

def atomic_torch(value,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    with open(tmp,"wb") as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def source_hashes():
    paths=[ROOT/"train.py",ROOT/"preflight.py",ROOT/"pilot_queue.py",ROOT/"summarize.py",
        ROOT/"pilot_objective.py",ROOT/"pilot_config.json",
        SRC/"dataset.py",SRC/"model_factory.py",SRC/"objective.py",
        SRC/"configs/mto_eta0.json",SRC/"data/hashes.json",SRC/"data/normalization.json",
        SRC/"data/raw_labels.npz",SRC/"data/dataset.npz",SRC/"data/splits.json",SRC/"runs/mto_eta0/best.pt"]
    paths += sorted((SRC/"frozen_reference").rglob("*.py"))
    return {str(p):sha(p) for p in paths}

def setup(cfg):
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]);torch.set_num_threads(cfg["num_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def make_model(data,cfg,source_cfg):
    model=build(source_cfg,data.stats)
    ck=torch.load(SRC/"runs/mto_eta0/best.pt",map_location="cpu",weights_only=False)
    assert ck["config"]==source_cfg and ck["epoch"]==33
    model.load_state_dict(ck["model"])
    return model.cuda()

def metrics(y,p):
    d=p-y;sst=np.square(y-y.mean()).sum()
    return {"count":int(y.size),"sse":float(np.square(d).sum()),
        "sst":float(sst),"r2":float(1-np.square(d).sum()/sst),
        "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}

@torch.no_grad()
def validate(model,data,cfg,mode,raw):
    model.eval();inds=data.parts["val"];pv=[];ev=[];trace=[];summed=np.zeros(5,dtype=np.float64);counts=np.zeros(3,dtype=np.float64)
    for a in range(0,len(inds),cfg["batch_size"]):
        idx=inds[a:a+cfg["batch_size"]];x,y=data.batch(idx)
        e,A=model(**x);v=terms((e,A),y,data.stats,mode)
        ne=int(y["mask_E"].sum());na=int((y["mask_E"]&y["mask_A"]).sum())
        nf=int((y["mask_E"]&y["mask_A"]&y["mask_f"]).sum())
        for j,key in enumerate(("energy","trace","weighted_trace","intensity")):
            summed[j]+=float(v[key])*(ne if j==0 else nf if j==3 and mode=="direct_f_matched" else na)
        counts += np.array([ne,na,nf])
        ee=e.cpu().numpy().astype(np.float64);tr=np.trace(A.cpu().numpy().astype(np.float64),axis1=-2,axis2=-1)
        ev.append(ee);trace.append(tr);pv.append(C_F*ee*tr)
    pred=np.concatenate(pv);ep=np.concatenate(ev);tr=np.concatenate(trace)
    truth=raw["f"][inds].astype(np.float64);et=raw["E"][inds].astype(np.float64)
    at=raw["A"][inds].astype(np.float64);derived=C_F*et*np.trace(at,axis1=-2,axis2=-1)
    m=raw["mask_E"][inds]&raw["mask_A"][inds]&raw["mask_f"][inds]
    assert m.all() and np.isfinite(pred).all()
    f=metrics(truth[m],pred[m])
    f["per_state"]=[metrics(truth[:,j],pred[:,j]) for j in range(10)]
    r={"raw_f":f,"derived_f":metrics(derived[m],pred[m]),
       "energy":metrics(et[raw["mask_E"][inds]],ep[raw["mask_E"][inds]]),
       "old_objective":float(summed[0]/counts[0]+summed[1]/counts[1]),
       "arm_objective":float(summed[0]/counts[0]+summed[3]/(counts[2] if mode=="direct_f_matched" else counts[1])),
       "weighted_trace":float(summed[2]/counts[1])}
    return r,{"ids":data.ids[inds],"indices":inds,"f_pred":pred,"f_true":truth,
              "E_pred":ep,"E_true":et,"f_derived_truth":derived}

def write_predictions(path,arr):
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**arr);os.replace(tmp,path)

def save_checkpoint(model,epoch,result,path,manifest):
    atomic_torch({"model":model.state_dict(),"epoch":epoch,"validation":result,
                  "source_manifest":manifest},path)

def run(arm):
    cfg=json.loads((ROOT/"pilot_config.json").read_text())
    assert arm in cfg["arms"]
    out=ROOT/"runs"/arm;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/"worker.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=source_hashes()
    pre=json.loads((ROOT/"preflight_results.json").read_text())
    assert pre["passed"] and pre["source_checkpoint_sha256"]==cfg["source_checkpoint_sha256"]
    assert pre["source_hashes"]==manifest, "Stale preflight or source changed"
    if (out/"FIT_COMPLETE.json").exists():
        fit=json.loads((out/"FIT_COMPLETE.json").read_text())
        assert fit["source_hashes"]==manifest and fit["arm"]==arm
        print("Already complete",arm,flush=True);return
    assert cfg["allow_test_evaluation"] is False
    assert torch.cuda.is_available()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    setup(cfg)
    assert manifest[str(SRC/"runs/mto_eta0/best.pt")]==cfg["source_checkpoint_sha256"]
    data=Data(device="cuda")
    assert set(data.parts)=={"train","val","test"}
    assert len(data.parts["train"])==120355 and len(data.parts["val"])==6686
    data.stats["mean_E2_train"]=cfg["mean_E2_train"]
    for key in ("sE2","sA2","mean_E2_train"):
        assert abs(data.stats[key]-cfg[key])<1e-9
    with np.load(SRC/"data/raw_labels.npz") as z:
        raw={k:z[k] for k in ("E","A","f","mask_E","mask_A","mask_f")}
    for part in ("train","val"):
        ix=data.parts[part]
        assert raw["mask_E"][ix].all() and raw["mask_A"][ix].all() and raw["mask_f"][ix].all()
    source_cfg=json.loads((SRC/"configs/mto_eta0.json").read_text())
    model=make_model(data,cfg,source_cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["lr"],amsgrad=True,weight_decay=cfg["weight_decay"])
    generator=np.random.default_rng(cfg["order_seed"])
    state={"epoch":1,"cursor":0,"order":None,"steps":0,"history":[],"best":{"f":None,"old":None,"arm":None},
           "best_epoch":{"f":None,"old":None,"arm":None},"train_sum":0.,"train_count":0}
    if (out/"run_manifest.json").exists() and not (out/"last.pt").exists():
        raise RuntimeError("Stale prior run output without resumable last.pt")
    if (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        assert ck["manifest"]==manifest and ck["arm"]==arm and ck["config"]==cfg
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);state=ck["state"]
        generator.bit_generator.state=ck["rng_numpy"]
        torch.set_rng_state(ck["rng_torch"].cpu());torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        random.setstate(ck["rng_python"]);np.random.set_state(ck["rng_numpy_global"])
        with open(out/"history.jsonl","w") as hf:
            for row in state["history"]:hf.write(json.dumps(row,allow_nan=False)+"\n")
    atomic_json({"arm":arm,"config":cfg,"source":str(SRC),"source_hashes":manifest,
                 "initial_epoch":state["epoch"],"gpu":torch.cuda.get_device_name(0),
                 "pid":os.getpid(),"time":time.time()},out/"run_manifest.json")
    last=time.monotonic()
    def checkpoint():
        nonlocal last
        atomic_torch({"arm":arm,"config":cfg,"manifest":manifest,"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),"state":state,
            "rng_numpy":generator.bit_generator.state,"rng_torch":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state(),"rng_python":random.getstate(),
            "rng_numpy_global":np.random.get_state()},out/"last.pt")
        last=time.monotonic()
    def sync_aliases():
        sel=out/"selections"
        for label,alias in (("f","best_f.pt"),("old","best_old_objective.pt"),
                            ("arm","best_arm_objective.pt")):
            ep=state["best_epoch"][label]
            if ep is None:continue
            src=sel/(f"{label}_epoch{ep:03d}.pt")
            assert src.exists()
            tmp=out/(alias+".tmp")
            if tmp.exists():tmp.unlink()
            os.link(src,tmp);os.replace(tmp,out/alias)
        ep=state["best_epoch"]["f"]
        if ep is not None:
            src=sel/(f"f_epoch{ep:03d}_val.npz")
            assert src.exists()
            tmp=out/"best_f_val.npz.tmp"
            if tmp.exists():tmp.unlink()
            os.link(src,tmp);os.replace(tmp,out/"best_f_val.npz")
    def record(epoch,order_hash=None,train_mean=None):
        result,arrays=validate(model,data,cfg,arm,raw)
        scores={"f":result["raw_f"]["sse"],"old":result["old_objective"],"arm":result["arm_objective"]}
        selections=out/"selections";selections.mkdir(exist_ok=True)
        for label in ("f","old","arm"):
            if state["best"][label] is None or scores[label]<state["best"][label]-1e-12:
                path=selections/(f"{label}_epoch{epoch:03d}.pt")
                save_checkpoint(model,epoch,result,path,manifest)
                if label=="f":write_predictions(selections/(f"f_epoch{epoch:03d}_val.npz"),arrays)
                state["best"][label]=scores[label];state["best_epoch"][label]=epoch
        row={"epoch":epoch,"validation":result,"scores":scores,
             "best":copy.deepcopy(state["best"]),"best_epoch":copy.deepcopy(state["best_epoch"]),
             "order_sha256":order_hash,"train_mean":train_mean,
             "steps":state["steps"],"time":time.time()}
        state["history"].append(row)
        atomic_json(row,out/"status.json")
        print(json.dumps({"arm":arm,"epoch":epoch,"f_r2":result["raw_f"]["r2"],
            "best_f_epoch":state["best_epoch"]["f"],"old":scores["old"],"arm_loss":scores["arm"]}),flush=True)
    if (out/"last.pt").exists():sync_aliases()
    def commit_epoch():
        checkpoint()
        with open(out/"history.jsonl","w") as hf:
            for row in state["history"]:hf.write(json.dumps(row,allow_nan=False)+"\n")
        sync_aliases()
    if state["epoch"]==1 and not state["history"]:
        record(0)
        initial=state["history"][-1]["validation"]
        assert abs(initial["raw_f"]["r2"]-0.4052941183410983)<2e-5, "epoch0 f R2 mismatch"
        assert abs(initial["energy"]["mae"]-0.09138826415013168)<2e-4, "epoch0 E MAE mismatch"
        commit_epoch()
    while state["epoch"]<=cfg["epochs"]:
        epoch=state["epoch"];start=time.monotonic()
        if state["order"] is None:
            state["order"]=generator.permutation(data.parts["train"])
            state["cursor"]=0;state["train_sum"]=0.;state["train_count"]=0
        order=state["order"]
        order_hash=hashlib.sha256(order.tobytes()).hexdigest()
        model.train()
        while state["cursor"]<len(order):
            idx=order[state["cursor"]:state["cursor"]+cfg["batch_size"]]
            x,y=data.batch(idx)
            optimizer.zero_grad(set_to_none=True)
            t=terms(model(**x),y,data.stats,arm)
            loss=t["total"]
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
            state["train_sum"]+=float(loss.detach())*len(idx);state["train_count"]+=len(idx)
            state["cursor"]+=len(idx);state["steps"]+=1
            if state["steps"]%100==0:
                atomic_json({"arm":arm,"event":"TRAIN","epoch":epoch,"cursor":state["cursor"],
                    "steps":state["steps"],"loss":float(loss.detach()),"grad_norm":float(norm),
                    "seconds_this_epoch":time.monotonic()-start,"time":time.time()},out/"status.json")
            if STOP or time.monotonic()-last>=540:
                checkpoint()
                if STOP:
                    atomic_json({"arm":arm,"event":"STOPPED_CHECKPOINTED","epoch":epoch,
                        "cursor":state["cursor"],"time":time.time()},out/"status.json")
                    return
        record(epoch,order_hash,state["train_sum"]/max(state["train_count"],1))
        state["epoch"]+=1;state["cursor"]=0;state["order"]=None
        commit_epoch()
        if STOP:return
    atomic_json({"arm":arm,"event":"FIT_COMPLETE","epochs":cfg["epochs"],
        "best":state["best"],"best_epoch":state["best_epoch"],
        "steps":state["steps"],"source_hashes":manifest,"time":time.time()},out/"FIT_COMPLETE.json")
    print("FIT_COMPLETE",arm,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("arm",choices=["control","weighted","direct_f_matched"])
    args=ap.parse_args()
    try:run(args.arm)
    except BlockingIOError:print("worker already active",flush=True)
    except BaseException as exc:
        out=ROOT/"runs"/args.arm;out.mkdir(parents=True,exist_ok=True)
        atomic_json({"type":type(exc).__name__,"error":str(exc),"time":time.time()},out/"FAILED.json")
        raise

