#!/usr/bin/env python3
"""Shared frozen-feature cache and metric utilities; train/validation only."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
ARCH=RESEARCH/"architecture"
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
RUNTIME=ROOT/"runtime"
C_F=2.0/(3.0*27.211386245988)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def atomic_json(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)
def atomic_npz(arrays,path):
    path=Path(path);tmp=path.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays);os.replace(tmp,path)
def config():
    return json.loads((ROOT/"config.json").read_text())
def pinned_sources():
    cfg=config()
    pre=json.loads((ARCH/"PREFLIGHT.json").read_text())
    review=json.loads((ARCH/"EXECUTION_REVIEW.json").read_text())
    integrity=json.loads((ARCH/"INDEPENDENT_COMPLETION_AUDIT.json").read_text())
    replay=json.loads((RESEARCH/"completion_receipts/FINAL20_REPLAY_RESULTS.json").read_text())
    assert pre["passed"] and review["passed"] and integrity["passed"]
    assert pre["source_hashes"]==review["source_hashes"]
    for p,h in pre["source_hashes"].items():assert sha(p)==h,(p,"source hash drift")
    assert sha(SRC/"runs/mto_eta0/best.pt")==cfg["source_eta0_checkpoint_sha256"]
    assert sha(ARCH/"initial/retained_residual.pt")==cfg["initial_residual_checkpoint_sha256"]
    assert replay["recomputed"]["primary_raw_native_f"]["emitted_final20"]["r2"]<.40529412
    assert not cfg["allow_test"]
    paths=[ROOT/"FROZEN_RESIDUAL_PROTOCOL.md",ROOT/"config.json",ROOT/"common.py",
           ARCH/"PREFLIGHT.json",ARCH/"EXECUTION_REVIEW.json",
           ARCH/"INDEPENDENT_COMPLETION_AUDIT.json",
           RESEARCH/"completion_receipts/FINAL20_REPLAY_RESULTS.json"]
    return {str(p):sha(p) for p in paths}
class Split:
    """Only train or validation rows are materialized; never construct a test batch."""
    def __init__(self,part):
        assert part in ("train","val")
        with np.load(SRC/"data/dataset.npz",allow_pickle=False) as z:
            self.indices=z[part].copy()
            self.ids=z["ids"][self.indices].copy()
            self.z=z["z"][self.indices].copy()
            self.pos=z["pos"][self.indices].copy()
            self.edge=z["edge"][self.indices].copy()
        if part=="train":
            with np.load(SRC/"data/raw_labels.npz",allow_pickle=False) as z:
                assert np.array_equal(z["ids"][self.indices],self.ids)
                self.f_true=z["f"][self.indices].astype(np.float64)
                self.E_true=z["E"][self.indices].astype(np.float64)
                self.mask_f=z["mask_f"][self.indices].copy()
                self.mask_E=z["mask_E"][self.indices].copy()
        else:
            with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
                assert np.array_equal(z["indices"],self.indices)
                assert np.array_equal(z["ids"],self.ids)
                self.f_true=z["f_true"].astype(np.float64)
                self.E_true=z["E_true"].astype(np.float64)
                self.mask_f=z["mask_f_true"].copy()
                self.mask_E=z["mask_E_true"].copy()
        assert len(self.indices)==config()["expected_molecules"][part]
        assert len(set(self.ids.tolist()))==len(self.ids)
        assert self.f_true.shape==(len(self.ids),10)==self.E_true.shape
        assert self.mask_f.shape==self.f_true.shape==self.mask_E.shape
        assert self.mask_f.all() and self.mask_E.all()
    def batch(self,a,b,device):
        import torch
        z=torch.from_numpy(self.z[a:b]);pos=torch.from_numpy(self.pos[a:b])
        raw=torch.from_numpy(self.edge[a:b]).long();n=b-a
        mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x={"z":z[mask],"pos":pos[mask],
           "batch":torch.arange(n)[:,None].expand_as(z)[mask],
           "n":n,"edge_index":edge}
        return {k:v.to(device) if hasattr(v,"to") else v for k,v in x.items()}
def setup_torch(seed=11):
    import random,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def source_model(device):
    import torch
    sys.path.insert(0,str(ARCH))
    from model import build
    stats=json.loads((ARCH/"stats.json").read_text())
    model=build("retained_residual",stats).to(device).eval()
    for p in model.base.parameters():p.requires_grad_(False)
    return model
def features(model,x):
    import torch
    h,t,E=model.features(**x)
    A=model.original_matrix(h,t)
    base32=C_F*E*A.diagonal(dim1=-2,dim2=-1).sum(-1)
    base64=C_F*E.double()*A.double().diagonal(dim1=-2,dim2=-1).sum(-1)
    return {"h":h,"E_pred":E,"base32":base32,"base64":base64}
def feature_cache(split,model,device,batch_size=64,health_guard=None):
    import torch
    cols={k:[] for k in ("h","E_pred","base32","base64")}
    start=time.monotonic()
    with torch.inference_mode():
        for a in range(0,len(split.indices),batch_size):
            if health_guard and a%6400==0:health_guard()
            d=features(model,split.batch(a,min(a+batch_size,len(split.indices)),device))
            for k,v in d.items():cols[k].append(v.cpu().numpy())
    out={k:np.concatenate(v,axis=0) for k,v in cols.items()}
    out.update({"indices":split.indices,"ids":split.ids,"f_true":split.f_true,
                "E_true":split.E_true,"mask_f":split.mask_f,"mask_E":split.mask_E})
    n=len(split.indices)
    assert out["h"].shape==(n,10,128) and out["h"].dtype==np.float32
    for k in ("E_pred","base32"):assert out[k].shape==(n,10) and out[k].dtype==np.float32
    assert out["base64"].shape==(n,10) and out["base64"].dtype==np.float64
    assert all(np.isfinite(v).all() for v in out.values() if np.issubdtype(v.dtype,np.floating))
    return out,time.monotonic()-start
def load_cache(part):
    assert part in ("train","val")
    path=RUNTIME/(f"cache_{part}.npz")
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def head_from_initial(device):
    import torch
    model=source_model("cpu")
    head=model.strength_head.to(device)
    assert sum(p.numel() for p in head.parameters())==4161
    assert all(p.requires_grad for p in head.parameters())
    assert torch.count_nonzero(head[-1].weight)==0 and torch.count_nonzero(head[-1].bias)==0
    return head
def native_predictions(head,h,base32,base64,scale):
    import torch
    delta=scale*head(h).squeeze(-1)
    return {"delta32":delta,"train_f32":(base32+delta).abs(),
            "signed64":base64+delta.double(),"native_f64":(base64+delta.double()).abs()}
def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    d=p-y;SSE=float(np.square(d).sum());SST=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":SSE,"sst":SST,"r2":1-SSE/SST,
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
def metrics(cache,pred,scale,train_var,energy_var,q90=None,q99=None):
    y=cache["f_true"]; f=pred["f"];E=cache["E_true"];Ep=cache["E_pred"]
    assert f.shape==y.shape and np.isfinite(f).all()
    result={"raw_f":metric(y,f),"energy":metric(E,Ep),
            "per_state":[metric(y[:,j],f[:,j]) for j in range(10)],
            "f_loss_native":float(np.square(f-y).mean()/train_var),
            "energy_loss":float(np.square(Ep-E).mean()/energy_var)}
    result["common_objective_native"]=result["energy_loss"]+result["f_loss_native"]
    if q90 is not None:
        bright={}
        for label,threshold in (("q90",q90),("q99",q99)):
            assert threshold is not None
            mask=y>=threshold
            bright[label]={"threshold":float(threshold),"count":int(mask.sum()),
                "bright_sse":float(np.square(f[mask]-y[mask]).sum()),
                "below_sse":float(np.square(f[~mask]-y[~mask]).sum())}
        base=cache["base64"]
        signed=pred["signed"];delta=pred["delta"]
        result["bright"]=bright
        result["residual"]={"negative_signed_count":int((signed<0).sum()),
            "zero_signed_count":int((signed==0).sum()),
            "mean_abs_delta":float(np.abs(delta).mean()),
            "delta_quantiles":np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
            "signed_min":float(signed.min()),"signed_max":float(signed.max()),
            "base_vs_frozen_sse":float(np.square(base-y).sum()),
            "corrected_minus_base_sse":float(np.square(f-y).sum()-np.square(base-y).sum())}
    return result
def cache_arrays_gpu(cache,device):
    import torch
    return {"h":torch.as_tensor(cache["h"],device=device),
            "base32":torch.as_tensor(cache["base32"],device=device),
            "base64":torch.as_tensor(cache["base64"],device=device),
            "f32":torch.as_tensor(cache["f_true"].astype(np.float32),device=device)}

