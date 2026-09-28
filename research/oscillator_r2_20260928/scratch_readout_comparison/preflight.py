#!/usr/bin/env python3
"""Bounded train/val identity, fresh-core, objective and resume preflight; no test inference."""
import copy,fcntl,hashlib,inspect,io,json,math,os,random,shutil,sys,tempfile,time
from pathlib import Path
import numpy as np,torch
from common import ROOT,SRC,source_seal,code_hashes,cfg,stats,setup,sha,save_json,save_torch,save_npz,objective,evaluate,lr_for_epoch,TrainValData,state_hash
from model import NativeDirect,MTODirect,make_pair,counts
from train import aliases,rng_pack,rng_restore
sys.path.insert(0,str(SRC))
from dataset import Data
def step(m,o,x,y,st):
    o.zero_grad(set_to_none=True)
    v=objective(m(**x),y,st);assert all(torch.isfinite(z) for z in v)
    v[0].backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True)
    o.step()
    return [float(z.detach()) for z in v],float(gn)
def state_diff(a,b):
    return max(float((a[k]-b[k]).abs().max()) for k in a if a[k].numel() and a[k].dtype.is_floating_point)
def main():
    gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu) and gpu in (2,4)
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    arch=ROOT.parent/"architecture";sys.path.insert(0,str(arch));import gpu_health
    before=gpu_health.snapshot(gpu);assert gpu_health.eligible(before)
    source=source_seal();code=code_hashes();c=cfg();st=stats();setup(11)
    data=TrainValData("cuda")
    assert len(data.parts["train"])==120355 and len(data.parts["val"])==6686
    assert not np.intersect1d(data.parts["train"],data.parts["val"]).size
    assert all(getattr(data,k).all().item() for k in ("mask_E","mask_A","mask_f"))
    assert abs(st["sE2"]-c["sE2"])<1e-15 and abs(st["sf2"]-c["sf2"])<1e-15
    # Validation labels/ID identity only, no validation model inference or selection.
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        ix=data.parts["val"]
        assert np.array_equal(z["indices"],ix) and np.array_equal(z["ids"],data.ids_at(ix))
        assert np.array_equal(z["f_true"],data.raw_at(ix,"f"))
        assert np.array_equal(z["E_true"],data.raw_at(ix,"E"))
        loc=data.lookup[ix]
        assert np.array_equal(z["mask_f_true"],data.mask_f[loc].cpu().numpy())
        assert np.array_equal(z["mask_E_true"],data.mask_E[loc].cpu().numpy())
    ix=data.parts["train"][:64];x,y=data.batch(ix)
    old=Data.__new__(Data);old.device="cuda"
    for k in ("z","pos","E","A","f","edge","mask_E","mask_A","mask_f"):
        setattr(old,k,getattr(data,k))
    xo,yo=Data.batch(old,data.lookup[ix])
    assert x.keys()==xo.keys() and y.keys()==yo.keys()
    for k in x:assert x[k]==xo[k] if k=="n" else torch.equal(x[k],xo[k]),k
    for k in y:assert torch.equal(y[k],yo[k]),k
    init=json.loads((ROOT/"INITIALIZATION.json").read_text())
    pair,_=make_pair(st)
    init_equality={};params={};latent={}
    for arm,m in pair.items():
        saved=torch.load(ROOT/"initial"/f"{arm}.pt",map_location="cpu",weights_only=False)
        assert state_hash(m.state_dict())==saved["state_sha256"]
        assert sha(ROOT/"initial"/f"{arm}.pt")==init["initialization"][arm]["initial_pt_sha256"]
        assert state_hash(m.state_dict())==init["initialization"][arm]["state_sha256"]
        assert counts(m)==init["initialization"][arm]["parameter_counts"]
        params[arm]=counts(m)
        m=m.cuda();m.eval()
        with torch.no_grad():
            pred=m(**x)
            e=np.asarray(pred["E"].cpu(),dtype=np.float64);f=np.asarray(pred["f"].cpu(),dtype=np.float64)
            assert e.shape==f.shape==(64,10) and (e>0).all() and (f>0).all()
            emx=float(np.max(np.abs(e-np.asarray(st["mu_E"])[None,:])))
            fmx=float(np.max(np.abs(f-np.asarray(st["mu_f"])[None,:])))
            assert emx<1e-5 and fmx<1e-6
            init_equality[arm]={"max_abs_E_mean":emx,"max_abs_f_mean":fmx}
    native_source=Path(inspect.getfile(type(pair["native"].base).__mro__[1])).resolve()
    mto_source=Path(inspect.getfile(type(pair["mto"].base.core).__mro__[1])).resolve()
    assert native_source==SRC/"official_detanet/detanet_model/detanet.py"
    assert mto_source==SRC/"frozen_reference/upstream/vendor/detanet_model/detanet.py"
    assert sha(native_source)==sha(mto_source)
    imported_core_sources={"native_official":str(native_source),"mto_vendored":str(mto_source),
        "source_sha256":sha(native_source)}
    assert torch.equal(pair["native"].base.mass,pair["mto"].base.core.mass)
    assert torch.equal(pair["native"].base.Embedding.elec,pair["mto"].base.core.Embedding.elec)
    # Hook the ACTUAL official native forward immediately after its last interaction block.
    observed=[]
    hook=pair["native"].base.blocks[-1].register_forward_hook(
        lambda _module,_args,output:observed.append(tuple(v.detach().clone() for v in output)))
    with torch.no_grad():
        _=pair["native"](**x)
        ms,mt=pair["mto"].base.core(z=x["z"],pos=x["pos"],batch=x["batch"],edge_index=x["edge_index"])
    hook.remove()
    assert len(observed)==1
    ns,nt=observed[0]
    ds=float((ns-ms).abs().max());dt=float((nt-mt).abs().max())
    assert torch.allclose(ns,ms,atol=5e-6,rtol=1e-5) and torch.allclose(nt,mt,atol=5e-6,rtol=1e-5),(ds,dt)
    latent={"scalar_max_abs":ds,"tensor_max_abs":dt}
    # Native and MTO use exact common optimizer/loss/order and required first-step gradient behavior.
    gradients={}
    for arm in ("native","mto"):
        m=pair[arm];m.train()
        opt=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,amsgrad=True)
        core=(m.base if arm=="native" else m.base.core)
        first_name=next(k for k,p in core.named_parameters() if p.ndim>1)
        first_p=dict(core.named_parameters())[first_name]
        opt.zero_grad(set_to_none=True)
        pred=m(**x);v=objective(pred,y,st)
        assert abs(float(v[0])-float(v[1]+v[2]))<1e-5
        v[0].backward()
        early=0. if first_p.grad is None else float(first_p.grad.abs().max())
        assert early==0.
        grad0=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True));opt.step()
        x2,y2=data.batch(data.parts["train"][64:128]);opt.zero_grad(set_to_none=True)
        pred2=m(**x2);v2=objective(pred2,y2,st);v2[0].backward()
        later=0. if first_p.grad is None else float(first_p.grad.abs().max())
        assert later>0. and math.isfinite(later),(arm,later)
        grad1=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True));opt.step()
        gradients[arm]={"first_core_grad_max":early,"second_core_grad_max":later,
            "grad_norm_first":grad0,"grad_norm_second":grad1}
    # Fixed nonzero terminal-head probe makes equivariance checks nonvacuous.
    for arm in ("native","mto"):
        with torch.no_grad():
            if arm=="native":
                terminal=[q for q in pair[arm].base.sout.modules() if isinstance(q,torch.nn.Linear)][-1]
                terminal.weight.fill_(.03/128)
                terminal.bias.zero_()
            else:
                pair[arm].base.decoder.energy_head.weight.fill_(.03/128)
                pair[arm].base.decoder.energy_head.bias.zero_()
                pair[arm].strength_head[-1].weight.fill_(.03/32)
                pair[arm].strength_head[-1].bias.zero_()
    # Rotation and atom-order invariance with nonconstant E/f on TRAIN molecules only.
    tx,_=data.batch(data.parts["train"][:2])
    torch.manual_seed(20260928);rotmat=torch.linalg.qr(torch.randn(3,3,device="cuda")).Q
    perm=torch.randperm(len(tx["z"]),device="cuda");invperm=torch.argsort(perm)
    xrot=dict(tx,pos=tx["pos"]@rotmat.T)
    xperm=dict(tx,z=tx["z"][perm],pos=tx["pos"][perm],
        batch=tx["batch"][perm],edge_index=invperm[tx["edge_index"]])
    invariance={}
    for arm in ("native","mto"):
        m=pair[arm].eval()
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            ref=m(**tx);pr=m(**xrot);pp=m(**xperm)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        vals={}
        for key in ("E","f"):
            spread=float((ref[key][0]-ref[key][1]).abs().max())
            assert spread>1e-7,(arm,key,spread)
            rerr=float((pr[key]-ref[key]).abs().max());perr=float((pp[key]-ref[key]).abs().max())
            assert torch.allclose(pr[key],ref[key],atol=2e-5,rtol=2e-4),(arm,key,rerr)
            assert torch.allclose(pp[key],ref[key],atol=2e-5,rtol=2e-4),(arm,key,perr)
            vals[key]={"between_molecule_same_state_span":spread,"rotation_max_abs":rerr,"permutation_max_abs":perr}
        invariance[arm]=vals
        m.train();assert m.training
    # Fresh initialization recreation and no pretrained loading.
    fresh,_=make_pair(st)
    assert all(state_hash(fresh[k].state_dict())==init["initialization"][k]["state_sha256"] for k in ("native","mto"))
    del fresh
    # Exact frozen NumPy order plan, LR boundary and update count.
    plan=json.loads((ROOT/"ORDER_PLAN.json").read_text())["orders"];assert len(plan)==100
    gen=np.random.default_rng(11)
    for ep,row in enumerate(plan,1):
        order=gen.permutation(data.parts["train"])
        assert hashlib.sha256(order.tobytes()).hexdigest()==row["sha256"] and row["updates"]==1881
        assert row["last_batch"]==35 and row["epoch"]==ep
    assert [lr_for_epoch(e) for e in (1,60,61,85,86,100)]==[.001,.001,.0003,.0003,.0001,.0001]
    # Two-step serialized model, optimizer, RNG and order/cursor equivalence for BOTH arms.
    resume={}
    for arm,cls in (("native",NativeDirect),("mto",MTODirect)):
        m=cls(st).cuda()
        m.load_state_dict(torch.load(ROOT/"initial"/f"{arm}.pt",map_location="cpu",weights_only=False)["model"])
        o=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,amsgrad=True)
        gen=np.random.default_rng(11);order=gen.permutation(data.parts["train"])
        b1=data.batch(order[:64]);b2=data.batch(order[64:128])
        step(m,o,*b1,st)
        record={"model":copy.deepcopy(m.state_dict()),"optimizer":copy.deepcopy(o.state_dict()),
            "rng":rng_pack(gen),"order":order.copy(),"cursor":64}
        buf=io.BytesIO();torch.save(record,buf);buf.seek(0)
        uninterrupted=step(m,o,*b2,st);final=copy.deepcopy(m.state_dict())
        saved=torch.load(buf,map_location="cuda",weights_only=False)
        r=cls(st).cuda();r.load_state_dict(saved["model"])
        ro=torch.optim.Adam(r.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,amsgrad=True)
        ro.load_state_dict(saved["optimizer"])
        gen2=np.random.default_rng(11);rng_restore(saved["rng"],gen2)
        assert saved["cursor"]==64 and np.array_equal(saved["order"],order)
        resumed=step(r,ro,*b2,st);rd=state_diff(final,r.state_dict())
        assert rd<=3e-6 and np.allclose(uninterrupted[0],resumed[0],rtol=1e-6,atol=1e-7),(arm,rd,uninterrupted,resumed)
        resume[arm]={"max_parameter_abs":rd,"second_losses":resumed[0],"second_grad_norm":resumed[1]}
    # Simulated interrupted selection must restore last committed epoch, not uncommitted alias.
    with tempfile.TemporaryDirectory(prefix="mto_scratch_selection_") as td:
        tmp=Path(td);(tmp/"selected").mkdir()
        rec={}
        for ep in (1,2):
            prefix=tmp/"selected"/f"raw_f_epoch{ep:03d}"
            save_torch({"epoch":ep},Path(str(prefix)+".pt"))
            save_npz(Path(str(prefix)+"_val.npz"),epoch=np.array([ep]))
            rec[ep]={"epoch":ep,"checkpoint_sha256":sha(Path(str(prefix)+".pt")),
                     "predictions_sha256":sha(Path(str(prefix)+"_val.npz"))}
        s={"best":{"raw_f":{"epoch":1}},"selection_artifacts":{"raw_f":rec[1]}}
        for label in ("joint",):
            prefix=tmp/"selected"/f"{label}_epoch001"
            save_torch({"epoch":1},Path(str(prefix)+".pt"));save_npz(Path(str(prefix)+"_val.npz"),epoch=np.array([1]))
            s["best"][label]={"epoch":1}
            s["selection_artifacts"][label]={"epoch":1,
                "checkpoint_sha256":sha(Path(str(prefix)+".pt")),
                "predictions_sha256":sha(Path(str(prefix)+"_val.npz"))}
        aliases(tmp,s)
        shutil.copy2(tmp/"selected/raw_f_epoch002.pt",tmp/"uncommitted.pt")
        shutil.copy2(tmp/"selected/raw_f_epoch002_val.npz",tmp/"uncommitted.npz")
        os.replace(tmp/"uncommitted.pt",tmp/"best_raw_f.pt")
        os.replace(tmp/"uncommitted.npz",tmp/"val_best_raw_f.npz")
        aliases(tmp,s)
        assert sha(tmp/"best_raw_f.pt")==rec[1]["checkpoint_sha256"]
        s["best"]["raw_f"]["epoch"]=2;s["selection_artifacts"]["raw_f"]=rec[2]
        aliases(tmp,s);assert sha(tmp/"best_raw_f.pt")==rec[2]["checkpoint_sha256"]
    after=gpu_health.snapshot(gpu)
    assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
    report={"passed":True,"classification":"Bounded train-only forward/grad/resume; validation label identity only, no validation model inference or test",
        "source_hashes":source,"code_hashes":code,"gpu":gpu,"health_before":before,
        "health_after":after,"parameters":params,"initial_mean_errors":init_equality,
        "common_core_latent":latent,"imported_core_sources":imported_core_sources,"gradients":gradients,"invariance":invariance,"two_step_resume":resume,"selection_crash_recovery":True,
        "order_hashes_all100_verified":True,"lr_boundary_verified":True,
        "raw_validation_identity_verified":True,"source_batch_adapter_exact":True,
        "test_batches":0,"time":time.time()}
    save_json(report,ROOT/"PREFLIGHT.json")
    print(json.dumps({"passed":True,"gpu":gpu,"latent":latent,"resume_max":max(v["max_parameter_abs"] for v in resume.values())}),flush=True)
if __name__=="__main__":main()

