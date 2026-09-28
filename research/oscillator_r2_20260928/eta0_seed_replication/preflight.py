#!/usr/bin/env python3
"""Bounded train-only legacy-equivalence, initialization, order and resume checks."""
import copy,fcntl,hashlib,io,json,os,random,shutil,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,SRC,ARCH,source_seal,code_hashes,config,setup,state_hash,save_json,save_torch,save_npz,sha,TrainValData,build,losses,C_F
from train import materialize_aliases
sys.path.insert(0,str(SRC))
from dataset import Data
def tensor_equal(a,b):
    assert a.keys()==b.keys()
    for k in a:
        assert torch.equal(a[k],b[k]),k
def max_state_difference(a,b):
    return max(float((u-v).abs().max()) for u,v in zip(a.values(),b.values()) if u.numel())
def forward_step(model,opt,x,y,stats):
    opt.zero_grad(set_to_none=True)
    value=losses(model(**x),y,stats,0.)
    value[0].backward()
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return tuple(float(z.detach()) for z in value),float(gn)
def main():
    gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and gpu in (1,6)
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH));import gpu_health
    before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
    source=source_seal();code=code_hashes()
    setup(23)
    data=TrainValData("cuda")
    plan=json.loads((ROOT/"ORDER_PLAN.json").read_text())
    assert len(data.parts["train"])==120355 and len(data.parts["val"])==6686
    assert plan["train_count"]==120355 and plan["updates_per_epoch"]==1881 and plan["last_batch"]==35
    assert len(np.intersect1d(data.parts["train"],data.parts["val"]))==0
    assert all(getattr(data,k).all().item() for k in ("mask_E","mask_A","mask_f"))
    # Label/identity alignment only: no validation model forward or metric selection.
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        ix=data.parts["val"]
        assert np.array_equal(z["indices"],ix)
        assert np.array_equal(z["ids"],data.ids_at(ix))
        assert np.array_equal(z["f_true"],data.raw_at(ix,"f"))
        assert np.array_equal(z["E_true"],data.raw_at(ix,"E"))
        local=data.lookup[ix]
        assert np.array_equal(z["mask_f_true"],data.mask_f[local].cpu().numpy())
        assert np.array_equal(z["mask_E_true"],data.mask_E[local].cpu().numpy())
    order_sample={}
    for seed in (23,37):
        rng=np.random.default_rng(seed);rows=plan["seeds"][str(seed)]
        assert len(rows)==100
        for e,row in enumerate(rows,1):
            order=rng.permutation(data.parts["train"])
            assert hashlib.sha256(order.tobytes()).hexdigest()==row["order_sha256"]
            assert e==row["epoch"] and np.array_equal(np.sort(order),np.sort(data.parts["train"]))
        order_sample[str(seed)]={"first":rows[0]["order_sha256"],"last":rows[-1]["order_sha256"]}
    assert order_sample["23"]["first"]!=order_sample["37"]["first"]
    # Original Data.batch method on the same TRAIN rows, remapped to compact local tensors.
    original=Data.__new__(Data)
    original.device="cuda"
    for key in ("z","pos","E","A","f","edge","mask_E","mask_A","mask_f"):
        setattr(original,key,getattr(data,key))
    indices=data.parts["train"][:64];local=data.lookup[indices]
    x,y=data.batch(indices);xo,yo=Data.batch(original,local)
    assert x.keys()==xo.keys() and y.keys()==yo.keys()
    for key in x:assert x[key]==xo[key] if key=="n" else torch.equal(x[key],xo[key])
    for key in y:assert torch.equal(y[key],yo[key])
    initial={}
    for seed in (11,23,37):
        c=config(23) if seed==11 else config(seed)
        if seed==11:c=dict(c,seed=11,name="mto_eta0")
        setup(seed);m1=build(c,data.stats).cuda();h1=state_hash(m1.state_dict())
        setup(seed);m2=build(c,data.stats).cuda();h2=state_hash(m2.state_dict())
        assert h1==h2
        assert sum(p.numel() for p in m1.parameters())==1552092
        assert sum(p.numel() for p in m1.core.parameters())==1371840
        assert c["energy_initialization"]=="train_state_means"
        with torch.no_grad():
            out=m1(**x)
            assert all(torch.isfinite(v).all() for v in out)
            e,a=out
            f=C_F*e.double()*a.double().diagonal(dim1=-2,dim2=-1).sum(-1)
            assert torch.isfinite(f).all() and (f>=0).all()
        initial[str(seed)]={"state_sha256":h1,"parameters":1552092,
            "backbone_parameters":1371840,"named_core_weight_sha256":
            hashlib.sha256(next(p for n,p in m1.named_parameters() if n.startswith("core.") and p.ndim>1)
                .detach().cpu().numpy().tobytes()).hexdigest()}
        if seed in (23,37):
            out=ROOT/f"runs/seed{seed}";out.mkdir(parents=True,exist_ok=True)
            path=out/"initial.pt"
            save_torch({"model":{k:v.detach().cpu().clone() for k,v in m1.state_dict().items()},
                "epoch":0,"initial_hash":h1,"seed":seed,"source_hashes":source,"code_hashes":code},path)
            initial[str(seed)]["initial_pt_sha256"]=sha(path)
        del m1,m2
    assert len({initial[str(i)]["state_sha256"] for i in (11,23,37)})==3
    assert len({initial[str(i)]["named_core_weight_sha256"] for i in (11,23,37)})==3
    archived=json.loads((SRC/"reports/initialization.json").read_text())["model_state_sha256"]
    assert initial["11"]["state_sha256"]==archived
    # Exact adapter/objective/gradient/one-step equality against source batch method.
    setup(23);c=config(23)
    base=build(c,data.stats).cuda()
    clone=build(c,data.stats).cuda();clone.load_state_dict(base.state_dict())
    v1=losses(base(**x),y,data.stats,0.)
    v2=losses(clone(**xo),yo,data.stats,0.)
    assert all(torch.equal(a,b) for a,b in zip(v1,v2))
    o1=torch.optim.Adam(base.parameters(),lr=.001,amsgrad=True)
    o2=torch.optim.Adam(clone.parameters(),lr=.001,amsgrad=True)
    vv1,g1=forward_step(base,o1,x,y,data.stats)
    vv2,g2=forward_step(clone,o2,xo,yo,data.stats)
    one_step_max=max_state_difference(base.state_dict(),clone.state_dict())
    one_step_loss_max=float(np.max(np.abs(np.asarray(vv1)-np.asarray(vv2))))
    one_step_grad_abs=abs(g1-g2)
    assert np.allclose(vv1,vv2,rtol=1e-6,atol=1e-6) and one_step_grad_abs<=1e-4 and one_step_max<=3e-6,(vv1,vv2,g1,g2,one_step_max)
    # Two-step uninterrupted versus exact serialized model/optimizer/scheduler/RNG/order/cursor.
    setup(23)
    u=build(c,data.stats).cuda()
    ou=torch.optim.Adam(u.parameters(),lr=.001,amsgrad=True)
    su=torch.optim.lr_scheduler.ReduceLROnPlateau(ou,factor=.5,patience=50,threshold=1e-4,threshold_mode="rel",min_lr=1e-6)
    gen=np.random.default_rng(23)
    order=gen.permutation(data.parts["train"])
    batches=[data.batch(order[a:a+64]) for a in (0,64)]
    forward_step(u,ou,*batches[0],data.stats)
    state={"model":copy.deepcopy(u.state_dict()),"opt":copy.deepcopy(ou.state_dict()),
           "scheduler":copy.deepcopy(su.state_dict()),"order":order.copy(),"cursor":64,
           "order_rng":copy.deepcopy(gen.bit_generator.state),"torch_cpu":torch.get_rng_state(),
           "torch_cuda":torch.cuda.get_rng_state(),"numpy_global":np.random.get_state(),
           "python":random.getstate()}
    buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
    v_uninterrupted=forward_step(u,ou,*batches[1],data.stats)
    final_u=copy.deepcopy(u.state_dict())
    saved=torch.load(buffer,map_location="cuda",weights_only=False)
    setup(23);r=build(c,data.stats).cuda()
    or_=torch.optim.Adam(r.parameters(),lr=.001,amsgrad=True)
    sr=torch.optim.lr_scheduler.ReduceLROnPlateau(or_,factor=.5,patience=50,threshold=1e-4,threshold_mode="rel",min_lr=1e-6)
    r.load_state_dict(saved["model"]);or_.load_state_dict(saved["opt"]);sr.load_state_dict(saved["scheduler"])
    gen2=np.random.default_rng(23);gen2.bit_generator.state=saved["order_rng"]
    torch.set_rng_state(saved["torch_cpu"].cpu());torch.cuda.set_rng_state(saved["torch_cuda"].cpu())
    np.random.set_state(saved["numpy_global"]);random.setstate(saved["python"])
    assert saved["cursor"]==64 and hashlib.sha256(saved["order"].tobytes()).hexdigest()==plan["seeds"]["23"][0]["order_sha256"]
    v_resumed=forward_step(r,or_,*batches[1],data.stats)
    resume_max=max_state_difference(final_u,r.state_dict())
    assert np.allclose(v_uninterrupted[0],v_resumed[0],rtol=1e-6,atol=1e-7) and abs(v_uninterrupted[1]-v_resumed[1])<=1e-5 and resume_max<=3e-6,(v_uninterrupted,v_resumed,resume_max)
    # Crash between writing a new selected artifact and committing last.pt:
    # the previously committed selection must remain reconstructible.
    with tempfile.TemporaryDirectory(prefix="mto_seed_selection_") as td:
        temp=Path(td);selected=temp/"selected";selected.mkdir()
        records={}
        for epoch in (1,2):
            pt=selected/f"legacy_epoch{epoch:03d}.pt"
            vals=selected/f"legacy_epoch{epoch:03d}_val.npz"
            save_torch({"epoch":epoch},pt)
            save_npz(vals,epoch=np.asarray([epoch]))
            records[epoch]={"epoch":epoch,"checkpoint_sha256":sha(pt),"predictions_sha256":sha(vals)}
        state1={"best_legacy_epoch":1,"best_raw_f_epoch":None,
                "selection_artifacts":{"legacy":records[1],"raw_f":None}}
        materialize_aliases(temp,state1)
        assert sha(temp/"best_legacy.pt")==records[1]["checkpoint_sha256"]
        shutil.copy2(selected/"legacy_epoch002.pt",temp/"best_legacy.uncommitted")
        shutil.copy2(selected/"legacy_epoch002_val.npz",temp/"val_best_legacy.uncommitted")
        os.replace(temp/"best_legacy.uncommitted",temp/"best_legacy.pt")
        os.replace(temp/"val_best_legacy.uncommitted",temp/"val_best_legacy.npz")
        assert sha(temp/"best_legacy.pt")==records[2]["checkpoint_sha256"]
        # Recover the old alias from the prior committed last.pt state.
        materialize_aliases(temp,state1)
        assert sha(temp/"best_legacy.pt")==records[1]["checkpoint_sha256"]
        assert sha(temp/"val_best_legacy.npz")==records[1]["predictions_sha256"]
        state2={"best_legacy_epoch":2,"best_raw_f_epoch":None,
                "selection_artifacts":{"legacy":records[2],"raw_f":None}}
        materialize_aliases(temp,state2)
        assert sha(temp/"best_legacy.pt")==records[2]["checkpoint_sha256"]
        assert sha(temp/"val_best_legacy.npz")==records[2]["predictions_sha256"]
    after=gpu_health.snapshot(gpu)
    assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
    report={"passed":True,"classification":"Fixed train-only batch, two-step optimizer resume and initialization preflight; no val/test inference",
        "time":time.time(),"source_hashes":source,"code_hashes":code,
        "initialization":initial,"order_first_last_hashes":order_sample,
        "validation_raw_label_identity_alignment_only":True,
        "adapter_matches_original_batch":True,"loss_gradient_update_match_with_fp32_cuda_tolerance":True,
        "one_step_max_parameter_difference":one_step_max,
        "one_step_loss_max_difference":one_step_loss_max,
        "one_step_grad_norm_absolute_difference":one_step_grad_abs,
        "two_step_resume_max_parameter_difference":resume_max,
        "interrupted_selection_alias_rollback_and_commit_pass":True,
        "numeric_tolerance_max_parameter_absolute":3e-6,
        "resume_second_loss":v_resumed,"initial_train_batch_native_f_finite":True,
        "gpu":gpu,"health_before":before,"health_after":after}
    save_json(report,ROOT/"PREFLIGHT.json")
    print(json.dumps({"passed":True,"seed23":initial["23"]["state_sha256"],
        "seed37":initial["37"]["state_sha256"],"gpu":gpu}),flush=True)
if __name__=="__main__":main()

