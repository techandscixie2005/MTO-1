#!/usr/bin/env python3
"""Validation-only inference of current channels64 best checkpoints; no test access."""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_chan64_20260928")
sys.path.insert(0,str(SRC))
from dataset import Data
from model_factory import build
OUT=Path(__file__).with_name("chan64_validation_only.json")
K=(2/3)/27.211386245988
torch.set_num_threads(2)
data=Data(device="cpu")
idx=data.parts["val"]
assert len(idx)==6686
raw=np.load(SRC/"data/raw_labels.npz")
truth=raw["f"][idx].astype(np.float64)
mask=raw["mask_f"][idx]&raw["mask_E"][idx]&raw["mask_A"][idx]
assert mask.all()
sse_base=float(np.sum((truth[mask]-truth[mask].mean())**2))
results={}
for name in ("G1","G2","G3","G4"):
    cfg=json.loads((SRC/"configs"/(name+".json")).read_text())
    path=SRC/"runs"/name/"best.pt"
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    ck=torch.load(path,map_location="cpu",weights_only=False)
    assert ck["config"]==cfg
    model=build(cfg,data.stats)
    model.load_state_dict(ck["model"])
    model.eval()
    energies=[]; traces=[]
    start=time.monotonic()
    with torch.no_grad():
        for a in range(0,len(idx),64):
            x,_=data.batch(idx[a:a+64])
            e,A=model(**x)
            energies.append(e.numpy())
            traces.append(A.diagonal(dim1=-2,dim2=-1).sum(-1).numpy())
    en=np.concatenate(energies).astype(np.float64)
    tr=np.concatenate(traces).astype(np.float64)
    e_for_f=en if cfg["supervise_E"] else raw["E"][idx].astype(np.float64)
    f=K*e_for_f*tr
    sse=float(np.sum((f[mask]-truth[mask])**2))
    results[name]={"checkpoint_sha256":digest,"checkpoint_epoch":ck["epoch"],
        "checkpoint_selection_val":ck["val"][0],
        "f_mode":"native" if cfg["supervise_E"] else "oracle_E_diagnostic_only",
        "val_f_r2":1-sse/sse_base,"val_f_sse":sse,
        "val_f_rmse":float(np.sqrt(np.mean((f[mask]-truth[mask])**2))),
        "elapsed_seconds":time.monotonic()-start}
    print(name,results[name],flush=True)
    del model
OUT.write_text(json.dumps({"classification":"validation-only audit of in-progress best checkpoints; G2/G4 E head unsupervised, so oracle-E f is diagnostic, not native performance","source":str(SRC),"results":results},indent=2))

