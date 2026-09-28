#!/usr/bin/env python3
"""Fixed scratch pair settings, train/val-only metrics and source seals."""
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from model import ROOT,SRC,make_pair,counts
from data_adapter import TrainValData
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def save_json(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,p)
def save_torch(obj,path):
    p=Path(path);tmp=p.with_suffix(".tmp")
    with open(tmp,"wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save_npz(path,**arrays):
    p=Path(path);tmp=p.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays);os.replace(tmp,p)
def cfg():return json.loads((ROOT/"config.json").read_text())
def stats():return json.loads((ROOT/"stats.json").read_text())
def source_seal():
    pins=json.loads((ROOT/"SOURCE_PINS.json").read_text())
    assert sha(ROOT/"PROTOCOL.md")==pins["protocol_sha256"]
    for path,digest in pins["paths"].items():assert sha(path)==digest,(path,"source drift")
    init=json.loads((ROOT/"INITIALIZATION.json").read_text())
    assert init["no_pretrained_checkpoint_loaded"] and init["no_val_test_labels_used"]
    assert sha(ROOT/"stats.json")==init["stats_sha256"]
    assert sha(ROOT/"ORDER_PLAN.json")==init["order_plan_sha256"]
    for arm,record in init["initialization"].items():
        assert sha(ROOT/"initial"/f"{arm}.pt")==record["initial_pt_sha256"]
    c=cfg()
    assert c["allow_test"] is False and c["allow_extra_epochs"] is False
    return {**pins["paths"],**{str(p):sha(p) for p in
        (ROOT/"PROTOCOL.md",ROOT/"SOURCE_PINS.json",ROOT/"config.json",
         ROOT/"stats.json",ROOT/"ORDER_PLAN.json",ROOT/"INITIALIZATION.json",
         ROOT/"initial/native.pt",ROOT/"initial/mto.pt")}}
def code_hashes():
    names=["model.py","data_adapter.py","prepare.py","common.py","train.py",
           "preflight.py","launch.py","summarize.py","config.json","PROTOCOL.md"]
    return {str(ROOT/n):sha(ROOT/n) for n in names}
def setup(seed=11):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def lr_for_epoch(epoch):
    assert 1<=epoch<=100
    return .001 if epoch<=60 else (.0003 if epoch<=85 else .0001)
def objective(pred,target,statistics):
    me=target["mask_E"];mf=target["mask_f"]
    E=pred["E"];f=pred["f"]
    assert E.shape==f.shape==me.shape==mf.shape
    le=(E[me]-target["E"][me]).square().mean()/statistics["sE2"]
    lf=(f[mf]-target["f"][mf]).square().mean()/statistics["sf2"]
    return le+lf,le,lf
def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    d=p-y;sse=float(np.square(d).sum());sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":sse,"sst":sst,"r2":1-sse/sst,
        "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
@torch.no_grad()
def evaluate(model,data,part,statistics,config):
    assert part in ("train","val")
    model.eval();indices=data.parts[part]
    fp=[];ep=[];ft=[];et=[];total=np.zeros(2,dtype=np.float64);counts=np.zeros(2,dtype=np.int64)
    for a in range(0,len(indices),config["batch_size"]):
        ix=indices[a:a+config["batch_size"]];x,y=data.batch(ix)
        pred=model(**x)
        v=objective(pred,y,statistics)
        ne=int(y["mask_E"].sum());nf=int(y["mask_f"].sum())
        total+=np.asarray([float(v[1])*ne,float(v[2])*nf]);counts+=np.asarray([ne,nf])
        ep.append(pred["E"].cpu().numpy())
        fp.append(pred["f"].cpu().numpy())
        et.append(data.raw_at(ix,"E"));ft.append(data.raw_at(ix,"f"))
    E=np.concatenate(ep);f=np.concatenate(fp);Et=np.concatenate(et);Ft=np.concatenate(ft)
    assert counts[0]==counts[1]==len(indices)*10 and E.shape==f.shape==(len(indices),10)
    assert np.isfinite(E).all() and np.isfinite(f).all() and (E>0).all() and (f>0).all()
    loss=[float(total[0]/counts[0]+total[1]/counts[1]),
        float(total[0]/counts[0]),float(total[1]/counts[1])]
    result={"raw_f":metric(Ft,f),"energy":metric(Et,E),
        "normalized_joint":loss,
        "per_state":[{"physical_state":j+1,**metric(Ft[:,j],f[:,j])} for j in range(10)],
        "tail":{},"negative_f_count":int((f<0).sum()),
        "zero_f_count":int((f==0).sum()),"negative_E_count":int((E<0).sum()),
        "max_f":float(f.max()),"mean_f_bias":float((f-Ft).mean())}
    for name,t in (("q90",config["q90"]),("q99",config["q99"])):
        mask=Ft>=t
        result["tail"][name]={"threshold":t,"bright_count":int(mask.sum()),
            "bright_sse":float(np.square(f[mask]-Ft[mask]).sum()),
            "below_count":int((~mask).sum()),
            "below_sse":float(np.square(f[~mask]-Ft[~mask]).sum())}
    molecule_sse=np.square(f-Ft).sum(1)
    ranking=np.argsort(-molecule_sse,kind="stable")
    result["molecule_error_concentration"]={name:{"molecules":n,
        "share_of_sse":float(molecule_sse[ranking[:n]].sum()/molecule_sse.sum())}
        for name,n in (("top1",1),("top10",10),
            ("top1pct",max(1,math.ceil(.01*len(indices)))))}
    if part=="val":
        pos=np.flatnonzero(data.ids_at(indices)==14562);assert len(pos)==1
        i=int(pos[0]);result["fixed_prior_case14562"]={
            "id":14562,"global_index":int(indices[i]),"true_f":Ft[i].tolist(),
            "predicted_f":f[i].tolist(),"molecule_sse":float(molecule_sse[i])}
    arrays={"ids":data.ids_at(indices),"indices":indices.copy(),
            "f_true":Ft,"E_true":Et,"f":f.astype(np.float64),"E":E.astype(np.float64)}
    return result,arrays

