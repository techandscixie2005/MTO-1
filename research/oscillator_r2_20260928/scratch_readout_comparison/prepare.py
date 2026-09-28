#!/usr/bin/env python3
"""Train-only raw-label means, paired fresh initialization and shared order plan."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np,torch
from model import ROOT,SRC,make_pair,counts
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write_json(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)
def save_pt(obj,path):
    p=Path(path);tmp=p.with_suffix(".tmp")
    with open(tmp,"wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def main():
    torch.set_num_threads(2);torch.manual_seed(11);np.random.seed(11)
    with np.load(SRC/"data/dataset.npz",allow_pickle=False) as z:
        train=z["train"].copy();val=z["val"].copy()
        assert len(train)==120355 and len(val)==6686
        assert np.intersect1d(train,val).size==0
    with np.load(SRC/"data/raw_labels.npz",allow_pickle=False) as z:
        E=z["E"][train].astype(np.float64);f=z["f"][train].astype(np.float64)
        me=z["mask_E"][train];mf=z["mask_f"][train]
    assert E.shape==f.shape==(120355,10) and me.all() and mf.all()
    assert np.isfinite(E).all() and np.isfinite(f).all()
    muE=E.mean(0);muf=f.mean(0)
    assert (muE>0).all() and (muf>0).all()
    sf2=float(np.square(f-f.mean()).mean());sf=float(np.sqrt(sf2))
    expected=.002510981243894732
    assert abs(sf2-expected)<=1e-12,(sf2,expected)
    source_stats=json.loads((SRC/"data/normalization.json").read_text())
    assert abs(source_stats["sE2"]-.5378066634062587)<1e-15
    stats={"mu_E":muE.tolist(),"mu_f":muf.tolist(),
        "sE2":source_stats["sE2"],"sf2":sf2,"sf":sf,
        "n_ref":source_stats["n_ref"],"E_state_mean":source_stats["E_state_mean"],
        "train_molecules":len(train),"states":10,
        "definition":"raw printed train-only label means; pooled population raw-f variance"}
    write_json(stats,ROOT/"stats.json")
    torch_rng_before_pair=torch.get_rng_state().tolist()
    models,mapping=make_pair(stats)
    torch_rng_after_pair=torch.get_rng_state().tolist()
    initial=ROOT/"initial";initial.mkdir(exist_ok=True)
    identities={}
    for arm,model in models.items():
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        digest=state_hash(state)
        save_pt({"arm":arm,"seed":11,"model":state,"state_sha256":digest,
                 "classification":"fresh scratch pair, no pretrained checkpoint"},initial/f"{arm}.pt")
        identities[arm]={"state_sha256":digest,"initial_pt_sha256":sha(initial/f"{arm}.pt"),
                          "parameter_counts":counts(model)}
    assert identities["native"]["state_sha256"]!=identities["mto"]["state_sha256"]
    native=models["native"].base.state_dict();mto=models["mto"].base.core.state_dict()
    for a,b in mapping.items():assert torch.equal(native[a],mto[b])
    order=[]
    rng=np.random.default_rng(11)
    sorted_train=np.sort(train)
    for epoch in range(1,101):
        ix=rng.permutation(train)
        assert np.array_equal(np.sort(ix),sorted_train)
        order.append({"epoch":epoch,"sha256":hashlib.sha256(ix.tobytes()).hexdigest(),
                      "molecules":len(ix),"updates":1881,"last_batch":35})
    write_json({"seed":11,"definition":"np.random.default_rng(11).permutation(original_train_indices) once/epoch",
                "epochs":100,"orders":order},ROOT/"ORDER_PLAN.json")
    info={"created":time.time(),"classification":"Fresh paired initialization and train-only moments",
        "stats_sha256":sha(ROOT/"stats.json"),"order_plan_sha256":sha(ROOT/"ORDER_PLAN.json"),
        "initialization":identities,"core_key_mapping":mapping,
        "common_core_key_count":len(mapping),"sf2_target_difference":sf2-expected,
        "torch_rng_before_pair_uint8":torch_rng_before_pair,
        "torch_rng_after_pair_uint8":torch_rng_after_pair,
        "readout_rng_context":"MTO factory and NativeDirect each call torch.manual_seed(11); final heads then zeroed",
        "torch_version":torch.__version__,"numpy_version":np.__version__,
        "no_pretrained_checkpoint_loaded":True,"no_val_test_labels_used":True}
    write_json(info,ROOT/"INITIALIZATION.json")
    print(json.dumps({"stats_sha256":info["stats_sha256"],"initial_native":identities["native"]["state_sha256"],
        "initial_mto":identities["mto"]["state_sha256"],"counts":{a:v["parameter_counts"] for a,v in identities.items()}}))
if __name__=="__main__":main()

