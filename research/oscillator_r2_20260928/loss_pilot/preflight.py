#!/usr/bin/env python3
"""Read-only CPU numerical/source preflight for the continuation screen."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(ROOT))
from train import Data,build,terms,C_F,source_hashes,sha,atomic_json
cfg=json.loads((ROOT/"pilot_config.json").read_text())
assert cfg["arms"]==["control","weighted","direct_f_matched"]
assert cfg["epochs"]==20 and cfg["selection"]=="validation_native_raw_f_SSE"
assert not cfg["allow_test_evaluation"]
assert sha(SRC/"runs/mto_eta0/best.pt")==cfg["source_checkpoint_sha256"]
pinned=json.loads((SRC/"data/hashes.json").read_text())
for name,expected in pinned.items():
    assert sha(SRC/"data"/name)==expected, "Pinned source data hash mismatch: "+name
with np.load(SRC/"data/dataset.npz") as dz, np.load(SRC/"data/raw_labels.npz") as rz:
    assert np.array_equal(dz["ids"],rz["ids"])
stats=json.loads((SRC/"data/normalization.json").read_text())
stats["mean_E2_train"]=cfg["mean_E2_train"]
for key in ("sE2","sA2","mean_E2_train"):
    assert abs(stats[key]-cfg[key])<1e-9
with np.load(SRC/"data/raw_labels.npz") as z:
    for key in ("mask_E","mask_A","mask_f"):assert z[key].all()
    raw_f=z["f"]
with np.load(SRC/"data/dataset.npz") as z:
    val_idx=z["val"];train_idx=z["train"];assert len(val_idx)==6686 and len(train_idx)==120355
    assert len(z["test"])==6686
with np.load(SRC/"runs/mto_eta0/val_predictions.npz") as z:
    y=z["f_true"];p=z["f"]
    assert np.array_equal(z["indices"],val_idx)
    assert np.allclose(y,raw_f[val_idx],rtol=0,atol=0)
    r2=1-np.square(y-p).sum()/np.square(y-y.mean()).sum()
    assert abs(r2-0.4052941183410983)<1e-10
source_cfg=json.loads((SRC/"configs/mto_eta0.json").read_text())
ck=torch.load(SRC/"runs/mto_eta0/best.pt",map_location="cpu",weights_only=False)
assert ck["config"]==source_cfg and ck["epoch"]==33
torch.set_num_threads(2)
data=Data(device="cpu")
sample=train_idx[:2]
x,y=data.batch(sample)
hashes=[];gradients={};forward=[]
for arm in cfg["arms"]:
    model=build(source_cfg,data.stats)
    model.load_state_dict(ck["model"])
    statehash=hashlib.sha256()
    for key,v in sorted(model.state_dict().items()):
        statehash.update(key.encode());statehash.update(v.detach().cpu().contiguous().numpy().tobytes())
    hashes.append(statehash.hexdigest())
    model.train()
    pred=model(**x)
    assert torch.isfinite(pred[0]).all() and torch.isfinite(pred[1]).all()
    forward.append(float(pred[0].detach().sum()))
    value=terms(pred,y,stats,arm)
    assert torch.isfinite(value["total"])
    value["total"].backward()
    grad=torch.stack([v.grad.abs().sum() for v in model.parameters() if v.grad is not None]).sum()
    assert torch.isfinite(grad) and grad>0
    gradients[arm]=float(grad)
    del model
assert len(set(hashes))==1
assert max(forward)-min(forward)<1e-12
orders=[]
for _ in cfg["arms"]:
    gen=np.random.default_rng(cfg["order_seed"])
    orders.append(hashlib.sha256(gen.permutation(train_idx).tobytes()).hexdigest())
assert len(set(orders))==1
# Analytic objective identity with true E and A-derived f.
e=torch.tensor([[3.,5.]],dtype=torch.float64,requires_grad=True)
a=torch.eye(3,dtype=torch.float64).reshape(1,1,3,3).repeat(1,2,1,1).requires_grad_()
f_derived=(C_F*e.detach()*a.detach().diagonal(dim1=-2,dim2=-1).sum(-1)).detach()
target={"E":e.detach().clone(),"A":a.detach().clone(),"f":f_derived,
        "mask_E":torch.ones((1,2),dtype=torch.bool),
        "mask_A":torch.ones((1,2),dtype=torch.bool),
        "mask_f":torch.ones((1,2),dtype=torch.bool)}
pred_e=(e.detach()+.2).requires_grad_()
pred_a=(a.detach()*1.1).requires_grad_()
v=terms((pred_e,pred_a),target,stats,"direct_f_matched")
v["total"].backward()
assert torch.isfinite(pred_e.grad).all() and pred_e.grad.abs().sum()>0
identity=terms((e.detach(),pred_a.detach()),target,stats,"direct_f_matched")
weighted=terms((e.detach(),pred_a.detach()),target,stats,"weighted")
assert abs(float(identity["intensity"]-weighted["intensity"]))<1e-12
# An actual model/optimizer step can be saved and resumed without changing its next update.
def model_and_optimizer():
    model=build(source_cfg,data.stats);model.load_state_dict(ck["model"])
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"],amsgrad=True,weight_decay=0.)
    return model,opt
def update(model,opt):
    opt.zero_grad(set_to_none=True)
    loss=terms(model(**x),y,stats,"control")["total"]
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"])
    opt.step()
one,opt_one=model_and_optimizer()
update(one,opt_one)
saved_model=copy.deepcopy(one.state_dict())
saved_opt=copy.deepcopy(opt_one.state_dict())
two,opt_two=model_and_optimizer()
two.load_state_dict(saved_model);opt_two.load_state_dict(saved_opt)
update(one,opt_one);update(two,opt_two)
resume_max_difference=max(float((one.state_dict()[k]-two.state_dict()[k]).abs().max())
                          for k in one.state_dict() if one.state_dict()[k].numel())
assert resume_max_difference<=1e-7
del one,two,opt_one,opt_two
manifest=source_hashes()
report={"passed":True,"time":time.time(),
        "source_checkpoint_sha256":cfg["source_checkpoint_sha256"],
        "source_hashes":manifest,
        "saved_validation_f_r2_reproduced":float(r2),
        "masks_all_valid":True,"sample_model_state_sha256":hashes[0],
        "sample_forward_energy_sum":forward[0],
        "sample_gradient_abs_sum_by_arm":gradients,
        "first_epoch_order_sha256":orders[0],
        "direct_f_energy_gradient_nonzero":True,
        "oracle_f_weighted_trace_identity":True,
        "resume_next_step_max_parameter_difference":resume_max_difference,
        "pinned_data_hashes_verified":pinned}
atomic_json(report,ROOT/"preflight_results.json")
print(json.dumps({"passed":True,"f_r2":r2,"gradients":gradients,
                  "source_files_hashed":len(manifest),
                  "first_epoch_order_sha256":orders[0]}))

