#!/usr/bin/env python3
"""Validation-only architecture screen with checkpointed 20-epoch arms."""
import argparse,copy,fcntl,hashlib,json,os,random,signal,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(SRC))
from dataset import Data
from model import build
from objective import terms,native_f64
from gpu_health import snapshot,unchanged
STOP=False
def request_stop(*_):
    global STOP
    STOP=True
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def atomic_json(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    os.replace(tmp,p)
def atomic_torch(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    with open(tmp,"wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def source_hashes():
    paths=[ROOT/n for n in ("train.py","preflight.py","architecture_queue.py","summarize.py",
        "gpu_health.py","model.py","objective.py","prepare.py","config.json","stats.json","PREPARATION.json",
        *(f"initial/{arm}.pt" for arm in json.loads((ROOT/"config.json").read_text())["arms"]))]
    paths += [SRC/n for n in ("dataset.py","model_factory.py","data/dataset.npz","data/raw_labels.npz",
        "data/splits.json","data/normalization.json","data/hashes.json","runs/mto_eta0/best.pt")]
    paths += [ROOT.parent/"resource_admission.py",
              ROOT.parent/"resource_audit/ADMISSION_REPORT.json"]
    paths += sorted((SRC/"frozen_reference").rglob("*.py"))
    return {str(p):sha(p) for p in paths}
def setup(cfg):
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]);torch.set_num_threads(cfg["num_threads"])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def metric(y,p):
    d=p-y;ss=float(np.square(d).sum());sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":ss,"sst":sst,"r2":1-ss/sst,
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
@torch.no_grad()
def validate(model,data,arm,cfg,raw,thresholds):
    model.eval();indices=data.parts["val"];fp=[];ep=[];signed_rows=[];delta_rows=[];loss_sum=0.;e_sum=0.;f_sum=0.;count=0
    for a in range(0,len(indices),cfg["batch_size"]):
        ix=indices[a:a+cfg["batch_size"]];x,y=data.batch(ix)
        pred=model(**x);value=terms(pred,y,data.stats)
        n=int(y["mask_f"].sum());ne=int(y["mask_E"].sum())
        assert n==ne==len(ix)*10
        loss_sum+=float(value["total"])*n;e_sum+=float(value["energy"])*n;f_sum+=float(value["intensity"])*n;count+=n
        fp.append(native_f64(pred,arm).cpu().numpy())
        if arm=="retained_residual":
            signed_rows.append(pred["signed_f"].double().cpu().numpy())
            delta_rows.append(pred["delta_f"].double().cpu().numpy())
        ep.append(pred["E"].double().cpu().numpy())
    f=np.concatenate(fp);e=np.concatenate(ep)
    yt=raw["f"][indices].astype(np.float64);et=raw["E"][indices].astype(np.float64)
    derived=(2/(3*27.211386245988))*et*np.trace(raw["A"][indices].astype(np.float64),axis1=-2,axis2=-1)
    assert np.isfinite(f).all() and f.shape==yt.shape==(6686,10)
    bright={}
    for label,t in thresholds.items():
        m=yt>=t
        bright[label]={"threshold":t,"count":int(m.sum()),
                       "sse":float(np.square(f[m]-yt[m]).sum())}
    result={"raw_f":metric(yt.reshape(-1),f.reshape(-1)),
        "derived_f":metric(derived.reshape(-1),f.reshape(-1)),
        "energy":metric(et.reshape(-1),e.reshape(-1)),
        "per_state":[metric(yt[:,j],f[:,j]) for j in range(10)],
        "bright":bright,"common_objective":loss_sum/count,
        "energy_loss":e_sum/count,"f_loss":f_sum/count}
    if arm=="retained_residual":
        signed=np.concatenate(signed_rows);delta=np.concatenate(delta_rows)
        result["residual_diagnostics"]={
            "signed_negative_count":int((signed<0).sum()),
            "signed_zero_count":int((signed==0).sum()),
            "final_zero_count":int((f==0).sum()),
            "delta_f_quantiles":np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
            "abs_delta_f_mean":float(np.abs(delta).mean()),
            "signed_f_min":float(signed.min()),
            "signed_f_max":float(signed.max())}
    arrays={"ids":data.ids[indices],"indices":indices,"f_pred":f,"f_true":yt,
            "E_pred":e,"E_true":et,"f_derived_truth":derived}
    return result,arrays
def save_npz(path,arrays):
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def run(arm):
    cfg=json.loads((ROOT/"config.json").read_text());assert arm in cfg["arms"]
    out=ROOT/"runs"/arm;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/"worker.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hashes=source_hashes()
    pre=json.loads((ROOT/"PREFLIGHT.json").read_text())
    review=json.loads((ROOT/"EXECUTION_REVIEW.json").read_text())
    assert pre["passed"] and pre["source_hashes"]==hashes
    assert review["passed"] and review["source_hashes"]==hashes
    assert not cfg["allow_test_evaluation"]
    if (out/"FIT_COMPLETE.json").exists():
        fit=json.loads((out/"FIT_COMPLETE.json").read_text())
        assert fit["source_hashes"]==hashes and fit["arm"]==arm
        print("Already complete",arm,flush=True);return
    if (out/"run_manifest.json").exists() and not (out/"last.pt").exists():
        raise RuntimeError("Stale output without resumable checkpoint")
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    physical=int(os.environ["MTO_PHYSICAL_GPU"])
    receipt=json.loads((out/"queue_launch_receipt.json").read_text())
    assert receipt["arm"]==arm and receipt["gpu"]==physical
    baseline=receipt["health_before"]
    assert snapshot(physical)["uuid"]==baseline["uuid"]
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    setup(cfg)
    data=Data(device="cuda")
    data.stats.update(json.loads((ROOT/"stats.json").read_text()))
    assert len(data.parts["train"])==120355 and len(data.parts["val"])==6686
    assert sha(SRC/"runs/mto_eta0/best.pt")==cfg["source_checkpoint_sha256"]
    with np.load(SRC/"data/raw_labels.npz") as z:
        raw={k:z[k] for k in ("E","A","f","mask_E","mask_A","mask_f")}
    for part in ("train","val"):
        ix=data.parts[part]
        assert all(raw[k][ix].all() for k in ("mask_E","mask_A","mask_f"))
    thresholds={"q90":float(np.quantile(raw["f"][data.parts["val"]],.9)),
                "q99":float(np.quantile(raw["f"][data.parts["val"]],.99))}
    model=build(arm,data.stats).cuda()
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=cfg["lr"],weight_decay=cfg["weight_decay"],amsgrad=True)
    generator=np.random.default_rng(cfg["order_seed"])
    state={"epoch":1,"cursor":0,"order":None,"steps":0,
           "train_sum":0.,"train_count":0,"history":[],
           "best":{"f":None,"loss":None},"best_epoch":{"f":None,"loss":None}}
    if (out/"last.pt").exists():
        ck=torch.load(out/"last.pt",map_location="cuda",weights_only=False)
        assert ck["hashes"]==hashes and ck["arm"]==arm and ck["config"]==cfg
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);state=ck["state"]
        generator.bit_generator.state=ck["rng_numpy"]
        torch.set_rng_state(ck["rng_torch"].cpu());torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        random.setstate(ck["rng_python"]);np.random.set_state(ck["rng_numpy_global"])
        with open(out/"history.jsonl","w") as hf:
            for row in state["history"]:hf.write(json.dumps(row,allow_nan=False)+"\n")
    atomic_json({"arm":arm,"config":cfg,"source_hashes":hashes,"physical_gpu":physical,
        "gpu_uuid":baseline["uuid"],"pid":os.getpid(),"initial_checkpoint_sha256":sha(ROOT/"initial"/(arm+".pt")),
        "parameter_counts":model.parameter_counts(),
        "thresholds":thresholds,"time":time.time()},out/"run_manifest.json")
    last=time.monotonic()
    def checkpoint():
        nonlocal last
        atomic_torch({"arm":arm,"config":cfg,"hashes":hashes,"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),"state":state,
            "rng_numpy":generator.bit_generator.state,"rng_torch":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state(),"rng_python":random.getstate(),
            "rng_numpy_global":np.random.get_state()},out/"last.pt")
        last=time.monotonic()
    def health_guard(epoch):
        now=snapshot(physical)
        if not unchanged(baseline,now):
            atomic_json({"event":"INVALID_GPU_HEALTH_CHANGE","epoch":epoch,
                         "baseline":baseline,"observed":now,"time":time.time()},out/"INVALID.json")
            raise RuntimeError("GPU ECC/remap changed; run invalid")
        return now
    def sync_aliases():
        sel=out/"selections"
        for label,alias in (("f","best_native_f.pt"),("loss","best_common_objective.pt")):
            ep=state["best_epoch"][label]
            if ep is None:continue
            src=sel/(f"{label}_epoch{ep:03d}.pt");assert src.exists()
            tmp=out/(alias+".tmp")
            if tmp.exists():tmp.unlink()
            os.link(src,tmp);os.replace(tmp,out/alias)
        ep=state["best_epoch"]["f"]
        if ep is not None:
            src=sel/(f"f_epoch{ep:03d}_val.npz");assert src.exists()
            tmp=out/"best_native_f_val.npz.tmp"
            if tmp.exists():tmp.unlink()
            os.link(src,tmp);os.replace(tmp,out/"best_native_f_val.npz")
    def record(epoch,order_hash=None,train_mean=None):
        health_before=health_guard(epoch)
        result,arrays=validate(model,data,arm,cfg,raw,thresholds)
        health_after=health_guard(epoch)
        scores={"f":result["raw_f"]["sse"],"loss":result["common_objective"]}
        sel=out/"selections";sel.mkdir(exist_ok=True)
        for label in ("f","loss"):
            if state["best"][label] is None or scores[label]<state["best"][label]-1e-12:
                path=sel/(f"{label}_epoch{epoch:03d}.pt")
                atomic_torch({"arm":arm,"epoch":epoch,"model":model.state_dict(),
                              "validation":result,"source_hashes":hashes},path)
                if label=="f":save_npz(sel/(f"f_epoch{epoch:03d}_val.npz"),arrays)
                state["best"][label]=scores[label];state["best_epoch"][label]=epoch
        row={"epoch":epoch,"validation":result,"scores":scores,
             "best":copy.deepcopy(state["best"]),"best_epoch":copy.deepcopy(state["best_epoch"]),
             "order_sha256":order_hash,"train_mean":train_mean,"steps":state["steps"],
             "gpu_health_before":health_before,"gpu_health_after":health_after,"time":time.time()}
        state["history"].append(row)
        atomic_json(row,out/"status.json")
        print(json.dumps({"arm":arm,"epoch":epoch,"f_r2":result["raw_f"]["r2"],
                          "best_f_epoch":state["best_epoch"]["f"],"gpu":physical}),flush=True)
    def commit():
        checkpoint()
        with open(out/"history.jsonl","w") as hf:
            for row in state["history"]:hf.write(json.dumps(row,allow_nan=False)+"\n")
        sync_aliases()
    if (out/"last.pt").exists():sync_aliases()
    if state["epoch"]==1 and not state["history"]:
        record(0)
        if arm in ("original","retained_residual"):
            r=state["history"][-1]["validation"]["raw_f"]["r2"]
            assert abs(r-0.4052941183410983)<2e-5,"original epoch0 baseline mismatch"
        commit()
    while state["epoch"]<=cfg["epochs"]:
        epoch=state["epoch"];start=time.monotonic()
        health_guard(epoch)
        if state["order"] is None:
            state["order"]=generator.permutation(data.parts["train"])
            state["cursor"]=0;state["train_sum"]=0.;state["train_count"]=0
        order=state["order"];order_hash=hashlib.sha256(order.tobytes()).hexdigest()
        model.train()
        while state["cursor"]<len(order):
            ix=order[state["cursor"]:state["cursor"]+cfg["batch_size"]]
            x,y=data.batch(ix)
            optimizer.zero_grad(set_to_none=True)
            value=terms(model(**x),y,data.stats);loss=value["total"]
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
            state["train_sum"]+=float(loss.detach())*len(ix);state["train_count"]+=len(ix)
            state["cursor"]+=len(ix);state["steps"]+=1
            if state["steps"]%100==0:
                atomic_json({"event":"TRAIN","arm":arm,"epoch":epoch,"cursor":state["cursor"],
                    "steps":state["steps"],"loss":float(loss.detach()),"grad_norm":float(norm),
                    "seconds_this_epoch":time.monotonic()-start,"time":time.time()},out/"status.json")
            if STOP or time.monotonic()-last>=540:
                checkpoint()
                if STOP:
                    atomic_json({"event":"STOPPED_CHECKPOINTED","arm":arm,"epoch":epoch,
                        "cursor":state["cursor"],"time":time.time()},out/"status.json")
                    return
        torch.cuda.synchronize()
        record(epoch,order_hash,state["train_sum"]/max(state["train_count"],1))
        state["epoch"]+=1;state["cursor"]=0;state["order"]=None
        commit()
        if STOP:return
    atomic_json({"event":"FIT_COMPLETE","arm":arm,"epochs":cfg["epochs"],
        "best":state["best"],"best_epoch":state["best_epoch"],
        "steps":state["steps"],"source_hashes":hashes,"time":time.time()},out/"FIT_COMPLETE.json")
    print("FIT_COMPLETE",arm,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("arm",choices=json.loads((ROOT/"config.json").read_text())["arms"])
    args=ap.parse_args()
    try:run(args.arm)
    except BlockingIOError:print("Worker already active",flush=True)
    except BaseException as exc:
        out=ROOT/"runs"/args.arm;out.mkdir(parents=True,exist_ok=True)
        atomic_json({"type":type(exc).__name__,"error":str(exc),"time":time.time()},out/"FAILED.json")
        raise

