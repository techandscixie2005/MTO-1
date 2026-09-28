#!/usr/bin/env python3
"""Frozen legacy eta0 settings, provenance and raw-native-f evaluation."""
import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
ARCH=ROOT.parent/"architecture"
sys.path.insert(0,str(SRC))
from model_factory import build
from objective import losses,oscillator,spectrum
from data_adapter import TrainValData
C_F=2.0/(3.0*27.211386245988)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def save_json(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)
def save_torch(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    with open(tmp,"wb") as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save_npz(path,**arrays):
    p=Path(path);tmp=p.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays);os.replace(tmp,p)
def config(seed):
    assert seed in (23,37)
    baseline=json.loads((SRC/"configs/mto_eta0.json").read_text())
    c=dict(baseline,seed=seed,name=f"eta0_seed{seed}",max_epochs=100,gpu=None)
    assert c["eta"]==0 and c["min_epochs"]==200 and c["early_patience"]==150
    return c
def source_seal():
    pins=json.loads((ROOT/"SOURCE_PINS.json").read_text())
    for path,digest in pins["paths"].items():assert sha(path)==digest,(path,"source drift")
    assert sha(ROOT/"PROTOCOL.md")==pins["protocol_sha256"]
    return {**pins["paths"],str(ROOT/"SOURCE_PINS.json"):sha(ROOT/"SOURCE_PINS.json"),
            str(ROOT/"PROTOCOL.md"):sha(ROOT/"PROTOCOL.md"),
            str(ROOT/"RESOURCE_GATE_AMENDMENT.md"):sha(ROOT/"RESOURCE_GATE_AMENDMENT.md")}
def code_hashes():
    names=("common.py","data_adapter.py","train.py","preflight.py","launch.py",
           "summarize.py","SOURCE_PINS.json","ORDER_PLAN.json","PROTOCOL.md")
    return {str(ROOT/n):sha(ROOT/n) for n in names}
def setup(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    d=p-y;sse=float(np.square(d).sum());sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":sse,"sst":sst,"r2":float(1-sse/sst),
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
@torch.no_grad()
def evaluate(model,data,part,c,with_spectrum=False):
    assert part in ("train","val")
    model.eval();indices=data.parts[part]
    E=[];A=[];F=[];Et=[];Ft=[];spec=np.zeros(2,dtype=np.float64);nspec=0
    sums=np.zeros(3,dtype=np.float64);counts=np.zeros(2,dtype=np.int64)
    for a in range(0,len(indices),64):
        ix=indices[a:a+64];x,y=data.batch(ix);pred=model(**x)
        vals=losses(pred,y,data.stats,c["eta"])
        ne=int(y["mask_E"].sum());na=int(y["mask_A"].sum())
        sums+=np.asarray([float(v) for v in vals[1:]])*np.asarray([ne,na,na])
        counts+=np.asarray([ne,na])
        e,am=pred
        en=e.cpu().numpy();an=am.cpu().numpy()
        native=C_F*en.astype(np.float64)*np.trace(an.astype(np.float64),axis1=-2,axis2=-1)
        E.append(en);A.append(an);F.append(native)
        Et.append(data.raw_at(ix,"E"));Ft.append(data.raw_at(ix,"f"))
        if with_spectrum:
            mask=y["mask_E"]&y["mask_f"]
            sp=spectrum(e,oscillator(pred),mask);st=spectrum(y["E"],y["f"],mask)
            d=(sp-st).double();spec+=np.asarray([float(d.square().sum()),float(d.abs().sum())])
            nspec+=d.numel()
    E=np.concatenate(E);A=np.concatenate(A);F=np.concatenate(F)
    Et=np.concatenate(Et);Ft=np.concatenate(Ft)
    assert np.isfinite(F).all() and np.isfinite(E).all() and counts[0]==counts[1]==len(indices)*10
    loss=[float(sums[0]/counts[0]+sums[1]/counts[1]+c["eta"]*sums[2]/counts[1]),
          float(sums[0]/counts[0]),float(sums[1]/counts[1]),float(sums[2]/counts[1])]
    result={"legacy_val_objective":loss,"raw_native_f":metric(Ft,F),
            "energy":metric(Et,E),
            "per_state":[{"state":j+1,**metric(Ft[:,j],F[:,j])} for j in range(10)],
            "tails":{}}
    for label,t in (("q90",.0546),("q99",.2377)):
        m=Ft>=t
        result["tails"][label]={"threshold":t,"bright_count":int(m.sum()),
             "bright_sse":float(np.square(F[m]-Ft[m]).sum()),
             "below_count":int((~m).sum()),"below_sse":float(np.square(F[~m]-Ft[~m]).sum())}
    if with_spectrum:
        result["spectrum"]={"MSE":float(spec[0]/nspec),"MAE":float(spec[1]/nspec)}
    arrays={"ids":data.ids_at(indices),"indices":indices.copy(),
            "f_true":Ft,"E_true":Et,"f":F,"E":E,"A":A}
    return result,arrays

