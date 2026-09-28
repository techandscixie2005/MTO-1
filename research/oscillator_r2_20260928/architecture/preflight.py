#!/usr/bin/env python3
"""CPU/source preflight for the four-arm validation-only architecture screen."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
sys.path.insert(0,str(ROOT))
from train import Data,build,terms,native_f64,source_hashes,sha,atomic_json
from model_factory import build as build_source
cfg=json.loads((ROOT/"config.json").read_text())
assert cfg["arms"]==["original","retained_residual","direct_f","independent_trace"]
assert cfg["epochs"]==20 and cfg["primary_metric"]=="pooled_raw_native_f_SSE"
assert cfg["shape_loss_weight"]==0 and not cfg["allow_test_evaluation"]
assert sha(SRC/"runs/mto_eta0/best.pt")==cfg["source_checkpoint_sha256"]
pinned=json.loads((SRC/"data/hashes.json").read_text())
for name,expected in pinned.items():
    assert sha(SRC/"data"/name)==expected, "Pinned dataset mismatch: "+name
prep=json.loads((ROOT/"PREPARATION.json").read_text())
assert prep["status"]=="prepared_no_joint_training"
assert prep["source_checkpoint_sha256"]==cfg["source_checkpoint_sha256"]
for label,name in (("config_sha256","config.json"),("stats_sha256","stats.json"),
                   ("subset_sha256","warmup_subset.json")):
    assert prep[label]==sha(ROOT/name),"Stale preparation metadata: "+name
for name in ("prepare.py","model.py","objective.py"):
    path=ROOT/name
    assert prep["source_hashes"][str(path)]==sha(path),"Stale preparation: "+name
for name in ("dataset.npz","raw_labels.npz"):
    path=SRC/"data"/name
    assert prep["source_hashes"][str(path)]==sha(path)
for arm in cfg["arms"]:
    assert sha(ROOT/"initial"/(arm+".pt"))==prep["arms"][arm]["initial_checkpoint_sha256"]
with np.load(SRC/"data/dataset.npz") as dz, np.load(SRC/"data/raw_labels.npz") as rz:
    assert np.array_equal(dz["ids"],rz["ids"])
    tr=dz["train"].copy();val=dz["val"].copy()
    assert len(tr)==120355 and len(val)==6686 and len(dz["test"])==6686
    assert len(set(tr)&set(val))==0
    subset=json.loads((ROOT/"warmup_subset.json").read_text())
    warm_idx=np.asarray(subset["indices"])
    assert len(warm_idx)==cfg["warmup"]["subset_molecules"]
    assert len(set(warm_idx.tolist()))==len(warm_idx)
    assert np.isin(warm_idx,tr).all()
    assert np.array_equal(dz["ids"][warm_idx],np.asarray(subset["molecule_ids"]))
    for name in ("mask_E","mask_A","mask_f"):assert rz[name].all()
    train_f=rz["f"][tr].astype(np.float64)
    train_trace=np.trace(rz["A"][tr].astype(np.float64),axis1=-2,axis2=-1)
stats=json.loads((ROOT/"stats.json").read_text())
assert stats["source"]=="train_only" and stats["normalization_scope"].startswith("f and trace")
assert abs(float(train_f.mean())-stats["f_mean_train"])<1e-10
assert abs(float(train_f.var())-stats["f_var_train"])<1e-10
assert abs(float(train_trace.mean())-stats["trace_mean_train"])<1e-10
assert abs(float(train_trace.std())-stats["trace_std_train"])<1e-10
normal=json.loads((SRC/"data/normalization.json").read_text())
assert stats["sE2"]==normal["sE2"]
with np.load(SRC/"runs/mto_eta0/val_predictions.npz") as z:
    assert np.array_equal(z["indices"],val)
    y=z["f_true"].astype(np.float64);p=z["f"].astype(np.float64)
    saved_r2=1-float(np.square(p-y).sum()/np.square(y-y.mean()).sum())
assert abs(saved_r2-0.4052941183410983)<1e-10
torch.set_num_threads(2)
torch.manual_seed(cfg["seed"])
data=Data(device="cpu");data.stats.update(stats)
x,target=data.batch(tr[:2])
source_cfg=json.loads((SRC/"configs/mto_eta0.json").read_text())
ck=torch.load(SRC/"runs/mto_eta0/best.pt",map_location="cpu",weights_only=False)
assert ck["config"]==source_cfg and ck["epoch"]==33
teacher=build_source(source_cfg,data.stats)
teacher.load_state_dict(ck["model"]);teacher.eval()
with torch.no_grad():
    src_e,src_a=teacher(**x)
    src_f=(2/(3*27.211386245988))*src_e.double()*src_a.double().diagonal(dim1=-2,dim2=-1).sum(-1)
reports={}
for arm in cfg["arms"]:
    model=build(arm,data.stats);model.eval()
    assert model.parameter_counts()==prep["arms"][arm]["parameter_counts"]
    with torch.no_grad():
        pred=model(**x);native=native_f64(pred,arm)
    assert native.shape==src_f.shape and torch.isfinite(native).all()
    assert torch.isfinite(pred["E"]).all()
    assert torch.allclose(pred["E"],src_e,atol=1e-6,rtol=1e-6)
    if arm in ("original","retained_residual"):
        assert torch.allclose(native,src_f,atol=1e-6,rtol=1e-6)
        assert torch.allclose(pred["A"],src_a,atol=1e-6,rtol=1e-6)
    if arm=="retained_residual":
        assert torch.count_nonzero(pred["delta_f"])==0
        assert torch.count_nonzero(pred["residual_delta"])==0
        assert pred["tensor_reconstruction_valid"].all()
        assert torch.allclose(pred["signed_f"].double(),src_f,atol=1e-6,rtol=1e-6)
    if arm=="direct_f":
        assert pred["A"] is None and pred["trace"] is None
        assert torch.all(native>=0)
    if arm=="independent_trace":
        assert torch.all(pred["trace"]>=0)
        assert torch.all(torch.linalg.eigvalsh(pred["A"])>=-1e-6)
        assert torch.allclose(pred["A"].diagonal(dim1=-2,dim2=-1).sum(-1),
                              pred["trace"],atol=1e-6,rtol=1e-6)
    model.train()
    value=terms(model(**x),target,data.stats)
    assert torch.isfinite(value["total"])
    value["total"].backward()
    grad=sum(float(v.grad.abs().sum()) for v in model.parameters() if v.grad is not None)
    assert np.isfinite(grad) and grad>0
    reports[arm]={"native_f_min":float(native.min()),"native_f_max":float(native.max()),
                  "gradient_abs_sum":grad,"parameter_counts":model.parameter_counts()}
    if arm in ("retained_residual","independent_trace"):
        model.zero_grad(set_to_none=True)
        terms(model(**x),target,data.stats)["intensity"].backward()
        if arm=="retained_residual":
            g=model.strength_head[-1].weight.grad
            assert g is not None and torch.isfinite(g).all() and float(g.abs().sum())>0
            for name in ("beta_head","tensor_gate"):
                grads=[p.grad for p in getattr(model.base.decoder,name).parameters()]
                assert any(g is not None and float(g.abs().sum())>0 for g in grads)
        else:
            grads=[p.grad for p in model.base.decoder.energy_head.parameters()]
            assert any(g is not None and float(g.abs().sum())>0 for g in grads)
    if arm=="direct_f":
        model.zero_grad(set_to_none=True)
        terms(model(**x),target,data.stats)["intensity"].backward()
        assert all(p.grad is None or float(p.grad.abs().sum())==0
                   for p in model.base.decoder.energy_head.parameters())
    del model
orders=[hashlib.sha256(np.random.default_rng(cfg["order_seed"]).permutation(tr).tobytes()).hexdigest()
        for _ in cfg["arms"]]
assert len(set(orders))==1
def fresh():
    m=build("original",data.stats)
    o=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),
                       lr=cfg["lr"],weight_decay=cfg["weight_decay"],amsgrad=True)
    return m,o
def step(m,o):
    o.zero_grad(set_to_none=True)
    loss=terms(m(**x),target,data.stats)["total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),cfg["grad_clip"])
    o.step()
one,opt_one=fresh();step(one,opt_one)
saved_model=copy.deepcopy(one.state_dict());saved_opt=copy.deepcopy(opt_one.state_dict())
two,opt_two=fresh();two.load_state_dict(saved_model);opt_two.load_state_dict(saved_opt)
step(one,opt_one);step(two,opt_two)
resume_max=max(float((one.state_dict()[k]-two.state_dict()[k]).abs().max())
               for k in one.state_dict() if one.state_dict()[k].numel())
assert resume_max<=1e-7
hashes=source_hashes()
report={"passed":True,"time":time.time(),"source_hashes":hashes,
        "source_checkpoint_sha256":cfg["source_checkpoint_sha256"],
        "pinned_dataset_hashes_verified":pinned,
        "raw_labels_sha256":sha(SRC/"data/raw_labels.npz"),
        "saved_validation_f_r2_reproduced":saved_r2,
        "initial_original_and_residual_f_exact_to_teacher_within_1e-6":True,
        "all_label_masks_valid":True,"train_only_f_variance_verified":True,
        "first_epoch_order_sha256":orders[0],"resume_max_parameter_difference":resume_max,
        "arm_checks":reports}
atomic_json(report,ROOT/"PREFLIGHT.json")
print(json.dumps({"passed":True,"arms":list(reports),"saved_val_f_r2":saved_r2,
                  "resume_max_difference":resume_max,"source_files_hashed":len(hashes)}))

