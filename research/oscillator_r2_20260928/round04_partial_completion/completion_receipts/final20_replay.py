#!/usr/bin/env python3
"""Deferred validation-only final20 replay. No training and no automatic scheduling."""
import argparse,fcntl,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARCH=ROOT.parent/"architecture"
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
RUN=ARCH/"runs/retained_residual"
PY=Path("/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python")
C_F=2/(3*27.211386245988)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1048576),b""):h.update(x)
    return h.hexdigest()
def save(obj,p):
    p=Path(p);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,p)
def metric(y,p):
    import numpy as np
    d=p-y;sst=float(np.square(y-y.mean()).sum())
    return {"count":int(y.size),"sse":float(np.square(d).sum()),"sst":sst,
            "r2":float(1-np.square(d).sum()/sst),
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
def summarize(y,eta0,base,delta,signed,final,ids,q90,e_true=None,e_pred=None):
    import numpy as np
    assert y.shape==eta0.shape==base.shape==delta.shape==signed.shape==final.shape
    assert y.ndim==2 and y.shape[1]==10 and len(ids)==len(y)
    assert all(np.isfinite(v).all() for v in (y,eta0,base,delta,signed,final))
    assert np.all(y>=0) and np.all(final>=0)
    assert np.allclose(final,np.abs(signed),rtol=0,atol=1e-7)
    identity_error=float(np.max(np.abs(signed-(base+delta))))
    assert identity_error<2e-6
    se0=np.square(eta0-y);seb=np.square(base-y);sef=np.square(final-y)
    total0=float(se0.sum());totalb=float(seb.sum());totalf=float(sef.sum())
    net=totalf-total0;via_base=totalb-total0;via_residual=totalf-totalb
    assert abs(net-(via_base+via_residual))<1e-9
    states=np.zeros_like(y,dtype=bool);states[:,6:8]=True
    weak=y<q90;negative=signed<0
    cells=[]
    for state_name,sm in (("S7_S8",states),("other_states",~states)):
        for strength_name,qm in (("below_q90",weak),("q90_or_brighter",~weak)):
            for sign_name,nm in (("negative_signed",negative),("nonnegative_signed",~negative)):
                mask=sm&qm&nm
                a=float(se0[mask].sum());b=float(seb[mask].sum());c=float(sef[mask].sum())
                cells.append({"state_group":state_name,"strength_group":strength_name,
                    "signed_group":sign_name,"count":int(mask.sum()),
                    "eta0_sse":a,"changed_base_sse":b,"emitted_final_sse":c,
                    "final_minus_eta0_sse":c-a,"base_minus_eta0_sse":b-a,
                    "final_minus_base_sse":c-b,
                    "share_of_net_sse_increase":(c-a)/net if net else None})
    assert sum(c["count"] for c in cells)==y.size
    for key,total in (("eta0_sse",total0),("changed_base_sse",totalb),
                      ("emitted_final_sse",totalf),("final_minus_eta0_sse",net)):
        assert abs(sum(c[key] for c in cells)-total)<1e-8
    per_state=[]
    for j in range(10):
        a=float(se0[:,j].sum());b=float(seb[:,j].sum());c=float(sef[:,j].sum())
        per_state.append({"physical_state":j+1,"eta0_sse":a,"changed_base_sse":b,
            "emitted_final_sse":c,"final_minus_eta0_sse":c-a,
            "base_minus_eta0_sse":b-a,"final_minus_base_sse":c-b})
    assert abs(sum(v["final_minus_eta0_sse"] for v in per_state)-net)<1e-8
    mol0=se0.sum(axis=1);molb=seb.sum(axis=1);molf=sef.sum(axis=1)
    moldelta=molf-mol0
    k_values=[("top1",1),("top5",min(5,len(y))),("top10",min(10,len(y))),
              ("top1pct",max(1,int(math.ceil(.01*len(y)))))]
    concentrations={}
    order_error=np.argsort(-molf,kind="stable")
    order_delta=np.argsort(-moldelta,kind="stable")
    for label,k in k_values:
        ie=order_error[:k];idelta=order_delta[:k]
        concentrations[label]={"molecules":k,
            "by_largest_final_error":{"share_of_final_sse":float(molf[ie].sum()/totalf) if totalf else None,
                "net_sse_change":float(moldelta[ie].sum()),
                "share_of_total_net_increase":float(moldelta[ie].sum()/net) if net else None},
            "by_largest_deterioration":{"net_sse_change":float(moldelta[idelta].sum()),
                "share_of_total_net_increase":float(moldelta[idelta].sum()/net) if net else None,
                "share_of_final_sse":float(molf[idelta].sum()/totalf) if totalf else None}}
    top_cases=[]
    for i in order_delta[:10]:
        top_cases.append({"id":int(ids[i]),"eta0_molecule_sse":float(mol0[i]),
                          "changed_base_molecule_sse":float(molb[i]),
                          "final_molecule_sse":float(molf[i]),
                          "final_minus_eta0_sse":float(moldelta[i])})
    neg_raw=(np.square(signed-y)-sef)[negative]
    expected=4*y[negative]*np.abs(signed[negative])
    assert np.allclose(neg_raw,expected,rtol=1e-7,atol=1e-7)
    result={"primary_raw_native_f":{"frozen_eta0":metric(y,eta0),
        "changed_base":metric(y,base),"emitted_final20":metric(y,final)},
        "sse_decomposition":{"net_final_minus_eta0":net,
            "changed_base_minus_eta0":via_base,
            "emitted_final_minus_changed_base":via_residual,
            "aggregate_identity_verified":True},
        "fixed_raw_f_q90":q90,"cross_table":cells,"per_state":per_state,
        "molecule_error_concentration":concentrations,
        "largest_deterioration_molecules":top_cases,
        "signed_diagnostics":{"negative_count":int(negative.sum()),
            "negative_fraction":float(negative.mean()),
            "zero_count":int((signed==0).sum()),
            "signed_min":float(signed.min()),"signed_max":float(signed.max()),
            "delta_min":float(delta.min()),"delta_max":float(delta.max()),
            "mean_abs_delta":float(np.abs(delta).mean()),
            "base_max":float(base.max()),
            "max_abs_signed_minus_base_plus_delta":identity_error,
            "abs_fold_improvement_in_sse_vs_unphysical_signed":float(neg_raw.sum()),
            "negative_fold_identity_verified":True}}
    if e_true is not None and e_pred is not None:
        result["energy"]=metric(e_true,e_pred)
    return result
class ValidationInputs:
    """Loads only val-indexed input rows, raw validation targets and masks."""
    def __init__(self):
        import numpy as np
        with np.load(SRC/"data/dataset.npz",allow_pickle=False) as z:
            self.indices=z["val"].copy();self.ids=z["ids"][self.indices].copy()
            self.z=z["z"][self.indices].copy()
            self.pos=z["pos"][self.indices].copy()
            self.edge=z["edge"][self.indices].copy()
        # The frozen validation export holds only validation targets and masks.
        # Avoid opening the full raw-label archive, which also contains test labels.
        with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
            assert np.array_equal(z["indices"],self.indices)
            assert np.array_equal(z["ids"],self.ids)
            self.f=z["f_true"].astype(np.float64)
            self.E=z["E_true"].astype(np.float64)
            for k in ("mask_E_true","mask_A_true","mask_f_true"):assert z[k].all()
        assert len(self.indices)==6686 and self.f.shape==(6686,10)
        assert len(set(self.ids.tolist()))==len(self.ids)
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
def self_test():
    import numpy as np
    protocol=json.loads((ROOT/"FINAL20_REPLAY_PROTOCOL.json").read_text())
    n=4;ids=np.arange(1,n+1);y=np.full((n,10),.02)
    y[0,6]=.15;y[1,7]=.0
    eta0=y+.01;base=y+.02
    delta=np.zeros_like(y);delta[0,6]=-.25;delta[1,7]=-.05
    signed=base+delta;final=np.abs(signed)
    r=summarize(y,eta0,base,delta,signed,final,ids,.0546)
    assert r["cross_table"] and r["sse_decomposition"]["aggregate_identity_verified"]
    assert any(c["count"]>0 for c in r["cross_table"] if c["state_group"]=="S7_S8"
               and c["strength_group"]=="below_q90" and c["signed_group"]=="negative_signed")
    v=ValidationInputs();x=v.batch(0,2,"cpu")
    assert x["n"]==2 and int(x["batch"].max())==1
    assert v.f.shape==(protocol["validation_molecules"],protocol["states_per_molecule"])
    assert float(np.quantile(v.f,.9))==protocol["raw_f_q90"]
    report={"passed":True,"time":time.time(),"script_sha256":sha(Path(__file__)),
        "protocol_sha256":sha(ROOT/"FINAL20_REPLAY_PROTOCOL.json"),
        "synthetic_decomposition_and_cross_table":True,
        "negative_fold_analytic_identity":True,
        "top1_5_10_1pct_counts":[r["molecule_error_concentration"][k]["molecules"]
                                 for k in ("top1","top5","top10","top1pct")],
        "validation_only_loader_first_two_molecules":True,
        "validation_q90":protocol["raw_f_q90"],"no_model_inference":True,
        "no_test_access":True}
    save(report,ROOT/"FINAL20_REPLAY_SELF_TEST.json")
    print(json.dumps(report))
def replay(args):
    import numpy as np
    auth_path=Path(args.authorization).resolve()
    auth=json.loads(auth_path.read_text())
    protocol_path=ROOT/"FINAL20_REPLAY_PROTOCOL.json"
    protocol=json.loads(protocol_path.read_text())
    assert auth.get("approved") is True and auth.get("scope")=="retained_residual_final20_validation_replay"
    gpu=int(auth["gpu"]);assert gpu in protocol["gpu_allowed"]
    assert auth["script_sha256"]==sha(Path(__file__))
    assert auth["protocol_sha256"]==sha(protocol_path)
    assert auth["final_checkpoint_sha256"]==protocol["expected_hashes"]["last.pt"]
    assert not (ROOT/"runtime/final20_val_replay.npz").exists(),"Replay already exists; do not repeat"
    assert not (ROOT/"FINAL20_REPLAY_RESULTS.json").exists()
    assert (ARCH/"ARCHITECTURE_RESULTS.json").exists(),"Matched architecture screen not summarized"
    cfg=json.loads((ARCH/"config.json").read_text())
    for arm in cfg["arms"]:
        assert (ARCH/"runs"/arm/"FIT_COMPLETE.json").exists(),"Matched screen incomplete: "+arm
    for name,digest in protocol["expected_hashes"].items():
        path=(RUN/name if name in ("FIT_COMPLETE.json","last.pt") else
              ARCH/name.removeprefix("architecture/") if name.startswith("architecture/") else
              SRC/"runs/mto_eta0/val_predictions.npz" if name=="eta0_val_predictions.npz" else
              SRC/"data"/name)
        assert sha(path)==digest,"Frozen input changed: "+name
    fit=json.loads((RUN/"FIT_COMPLETE.json").read_text())
    pre=json.loads((ARCH/"PREFLIGHT.json").read_text())
    review=json.loads((ARCH/"EXECUTION_REVIEW.json").read_text())
    assert fit["event"]=="FIT_COMPLETE" and fit["epochs"]==20 and fit["best_epoch"]["f"]==0
    assert pre["passed"] and review["passed"] and fit["source_hashes"]==pre["source_hashes"]==review["source_hashes"]
    for path,digest in fit["source_hashes"].items():assert sha(path)==digest
    assert not (RUN/"INVALID.json").exists() and not (RUN/"FAILED.json").exists()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2"
    os.environ["PYTHONWARNINGS"]="ignore"
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH))
    from gpu_health import snapshot,eligible,unchanged
    before=snapshot(gpu)
    assert eligible(before,guarded=(gpu==5)),"GPU unhealthy or occupied"
    micro=None
    if gpu==5:
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1")
        check=subprocess.run([str(PY),str(ROOT.parent/"resource_admission.py"),"--microcheck"],
                             env=env,text=True,capture_output=True,timeout=120)
        lines=[v for v in check.stdout.splitlines() if v.startswith("{")]
        micro=json.loads(lines[-1]) if check.returncode==0 and lines else {"passed":False,
            "exit_code":check.returncode,"stderr_tail":check.stderr[-1500:]}
        assert micro.get("passed") is True,"GPU5 numerical microcheck failed"
        checked=snapshot(gpu)
        assert unchanged(before,checked) and eligible(checked,guarded=True)
        before=checked
    import torch
    from model import build
    from objective import native_f64
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    val=ValidationInputs()
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        assert np.array_equal(z["ids"],val.ids) and np.array_equal(z["indices"],val.indices)
        assert np.array_equal(z["f_true"],val.f) and np.array_equal(z["E_true"],val.E)
        eta0=z["f"].astype(np.float64)
    stats=json.loads((ARCH/"stats.json").read_text())
    ck=torch.load(RUN/"last.pt",map_location="cpu",weights_only=False)
    assert ck["arm"]=="retained_residual" and ck["state"]["epoch"]==21 and ck["state"]["cursor"]==0
    assert ck["state"]["history"][-1]["epoch"]==20 and ck["hashes"]==fit["source_hashes"]
    model=build("retained_residual",stats).cuda().eval()
    model.load_state_dict(ck["model"])
    cols={k:[] for k in ("E_pred","base_f_float32","base_f_native64","delta_f","signed_f_float32","signed_f_native64","final_f")}
    with torch.inference_mode():
        for a in range(0,len(val.ids),64):
            b=min(a+64,len(val.ids))
            if a%640==0:
                now=snapshot(gpu)
                assert unchanged(before,now) and set(now["apps"]).issubset({os.getpid()}),"GPU fault/foreign process"
            pred=model(**val.batch(a,b,"cuda"))
            assert pred["tensor_reconstruction_valid"].all()
            E=pred["E"]
            raw=pred["base_A"]
            tr=raw.diagonal(dim1=-2,dim2=-1).sum(-1)
            base32=C_F*E*tr
            signed=pred["signed_f"];delta=pred["delta_f"]
            final=native_f64(pred,"retained_residual")
            base64=C_F*E.double()*raw.double().diagonal(dim1=-2,dim2=-1).sum(-1)
            signed64=base64+delta.double()
            assert torch.allclose(signed,base32+delta,atol=2e-6,rtol=0)
            assert torch.equal(final,signed64.abs())
            out={"E_pred":E.double(),"base_f_float32":base32.double(),
                 "base_f_native64":base64,"delta_f":delta.double(),
                 "signed_f_float32":signed.double(),"signed_f_native64":signed64,
                 "final_f":final}
            for key,tensor in out.items():cols[key].append(tensor.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in cols.items()}
    assert np.max(np.abs(arrays["signed_f_float32"]-arrays["signed_f_native64"]))<2e-6
    assert all(x.shape==(6686,10) and np.isfinite(x).all() for x in arrays.values())
    after=snapshot(gpu)
    assert unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
    summary=summarize(val.f,eta0,arrays["base_f_native64"],arrays["delta_f"],
                      arrays["signed_f_native64"],arrays["final_f"],val.ids,
                      protocol["raw_f_q90"],val.E,arrays["E_pred"])
    stored=ck["state"]["history"][-1]["validation"]
    assert abs(summary["primary_raw_native_f"]["emitted_final20"]["r2"]-stored["raw_f"]["r2"])<2e-5
    assert abs(summary["energy"]["mae"]-stored["energy"]["mae"])<2e-5
    runtime=ROOT/"runtime";runtime.mkdir(exist_ok=True)
    array_path=runtime/"final20_val_replay.npz"
    tmp=runtime/"final20_val_replay.tmp.npz"
    np.savez_compressed(tmp,ids=val.ids,indices=val.indices,f_true=val.f,E_true=val.E,
                        eta0_f=eta0,**arrays)
    os.replace(tmp,array_path)
    result={"classification":"Authorized one-time validation-only final20 diagnostic; selected best epoch0 unchanged",
        "time":time.time(),"authorization_sha256":sha(auth_path),
        "protocol_sha256":sha(protocol_path),"script_sha256":sha(Path(__file__)),
        "frozen_input_hashes":protocol["expected_hashes"],"gpu":gpu,"gpu_uuid":before["uuid"],
        "gpu_before":before,"gpu_after":after,"gpu5_microcheck":micro,
        "runtime_array_path":str(array_path),"runtime_array_sha256":sha(array_path),
        "runtime_arrays_excluded_from_archive":True,"full_validation_alignment":True,
        "stored_final20_r2":stored["raw_f"]["r2"],
        "stored_final20_energy_mae":stored["energy"]["mae"],
        "recomputed":summary,
        "limits":["Validation replay only; does not alter checkpoint selection or establish a matched-control advantage.",
                  "True raw f defines diagnostic bins, never model input. No test indices, labels, or predictions were read.",
                  "Base-versus-final errors are an inference decomposition of this jointly trained checkpoint, not a retraining counterfactual."]}
    save(result,ROOT/"FINAL20_REPLAY_RESULTS.json")
    print(json.dumps({"completed":True,"gpu":gpu,"final_r2":summary["primary_raw_native_f"]["emitted_final20"]["r2"],
                      "runtime_array_sha256":result["runtime_array_sha256"],
                      "summary":str(ROOT/"FINAL20_REPLAY_RESULTS.json")}))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("self-test","replay"))
    ap.add_argument("--authorization",help="Explicit root-approved JSON; required only for replay")
    args=ap.parse_args()
    if args.mode=="self-test":self_test()
    else:
        assert args.authorization,"Replay requires explicit authorization JSON"
        replay(args)
if __name__=="__main__":main()

