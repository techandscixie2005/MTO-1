#!/usr/bin/env python3
"""Frozen F5 historical-test evaluator. Preflight never opens a test array."""
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

BASE=Path("/home/inspur/MTO-1")
ROOT=BASE/"research/oscillator_r2_20260928"
HERE=ROOT/"final_f5_historical_test"
FREEZE=HERE/"CANDIDATE_FREEZE.json"
PROTOCOL=HERE/"PROTOCOL.md"
NAMES=("mto_eta0","mto_eta01","mto_eta1","G1","G3")
K=(2.0/3.0)/27.211386245988
GPU_UUID="GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184"
GEOMETRY_CODE_SHA="cda3735529a57e83a6183ef82a85fc022022517404681bff0635a9bf91e54472"
GPU_HEALTH_SHA="c0f82c09ed865361c055fee290ea3a2339c1fcf38c55cf92d3fb0dbd780afa17"

def need(v,msg):
    if not v: raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
def J(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def once(path,obj):
    path=Path(path);payload=(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as f:f.write(payload);f.flush();os.fsync(f.fileno())
def metric(y,p):
    need(y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all(),"metric shape/nonfinite")
    e=np.asarray(p-y,dtype=np.float64);yy=np.asarray(y,dtype=np.float64)
    sse=float(np.square(e).sum(dtype=np.float64))
    sst=float(np.square(yy-yy.mean(dtype=np.float64)).sum(dtype=np.float64))
    need(sst>0,"zero SST")
    return {"count":int(y.size),"sse":sse,"sst":sst,"r2":float(1-sse/sst),
            "mae":float(np.abs(e).mean(dtype=np.float64)),
            "rmse":float(np.sqrt(sse/y.size)),
            "mean_signed_error":float(e.mean(dtype=np.float64)),
            "overprediction_fraction":float((e>0).mean())}
def aligned(arrays,names=NAMES):
    a=arrays[names[0]]
    for name in names:
        z=arrays[name]
        for key in ("ids","indices","f_true","mask_f_true"):
            need(np.array_equal(a[key],z[key]),"array alignment "+name+" "+key)
        need(z["f_true"].dtype==np.float64 and z["mask_f_true"].dtype==np.bool_,
             "raw truth/mask dtype "+name)
        need(z["f"].shape==z["f_true"].shape and np.isfinite(z["f"]).all(),
             "native prediction shape/finiteness "+name)
    need(a["mask_f_true"].all(),"unexpected invalid raw-f labels")
    return a
def load_pred(path,expected):
    need(path.is_file() and sha(path)==expected,"saved prediction hash "+str(path))
    with np.load(path,allow_pickle=False) as z:
        keys=("ids","indices","f_true","mask_f_true","f")
        need(all(k in z.files for k in keys),"saved prediction keys "+str(path))
        out={k:z[k].copy() for k in keys}
    return out
def verify_static(f):
    need(f["candidate"]=="F5" and f["component_order"]==list(NAMES),"frozen candidate/order")
    need(f["validation_r2"]==0.48769960489832564,"frozen validation identity")
    need(f["prospective_diagnostics"]["raw_f_tail_thresholds"]=={"q90":0.0549,"q99":0.2412},
         "TRAIN tail thresholds")
    need(f["prospective_diagnostics"]["q2_boundaries"]==[1e-5,0.05941956341810567,
         0.09867099135475822,0.21033971729099188,0.45009813312285296],"TRAIN q2 cuts")
    for table in ("source_data_scaler_hashes_verified","artifact_and_provenance_hashes_verified"):
        need(all("test_predictions.npz" not in path for path in f[table]),
             "test array in preparation seal")
        verify_hash_map(f[table])
    need(sha(ROOT/"geometry_error_audit/geometry_error_audit.py")==GEOMETRY_CODE_SHA,
         "geometry code drift")
    need(sha(ROOT/"architecture/gpu_health.py")==GPU_HEALTH_SHA,"GPU health source drift")
    import torch
    for name in NAMES:
        c=f["components"][name]
        checkpoint=torch.load(c["selected_checkpoint_path"],map_location="cpu",weights_only=False)
        need(checkpoint["epoch"]==c["epoch"] and checkpoint["config"]==c["config"] and
             isinstance(checkpoint["model"],dict) and checkpoint["model"],
             "CPU checkpoint metadata "+name)
        del checkpoint
        need(c["weight_numerator"]==1 and c["weight_denominator"]==5,"component weight "+name)
        need(sha(c["selected_checkpoint_path"])==c["checkpoint_sha256"],"checkpoint drift "+name)
        need(sha(c["validation_reference"]["prediction_path"])==
             c["validation_reference"]["prediction_sha256"],"validation reference drift "+name)
        need(sha(c["run_manifest_path"])==c["run_manifest_sha256"],"run manifest drift "+name)
    need(sha(ROOT/"TAIL_THRESHOLD_PROVENANCE_ERRATUM.json")==
         f["prospective_diagnostics"]["tail_erratum_sha256"],"TRAIN tail erratum drift")
def pinned_geometry():
    need(sha(ROOT/"geometry_error_audit/geometry_error_audit.py")==GEOMETRY_CODE_SHA,
         "frozen geometry helper drift")
    from importlib.util import spec_from_file_location,module_from_spec
    path=ROOT/"geometry_error_audit/geometry_error_audit.py"
    spec=spec_from_file_location("frozen_geometry",path)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module

def fixed_input_batch(z_np,pos_np,edge_np,torch):
    # Same arithmetic, tensor casts, and edge order as original Data.batch,
    # applied only to explicitly selected rows.
    n=len(z_np)
    z=torch.from_numpy(np.array(z_np,copy=True))
    pos=torch.from_numpy(np.array(pos_np,copy=True))
    raw=torch.from_numpy(np.array(edge_np,copy=True))
    mask=z!=0;counts=mask.sum(1)
    offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
    edge_mask=raw[:,0]>=0
    edge=(raw.long()+offsets[:,None,None]).transpose(1,2)[edge_mask].T.contiguous()
    return {"z":z[mask],"pos":pos[mask],
            "batch":torch.arange(n)[:,None].expand_as(z)[mask],
            "n":n,"edge_index":edge}

def verify_input_adapter_validation():
    # Test the real original Data.batch method on validation-only buffers
    # without calling its eager constructor or reading any test row/label.
    import torch
    geo=pinned_geometry()
    src=BASE/"experiments/qm9s_chan64_20260928"
    dataset=src/"data/dataset.npz"
    val=np.asarray(geo.npz_member_memmap(dataset,"val"),dtype=np.int64)
    need(val.shape==(6686,),"validation input population")
    sys.path.insert(0,str(src))
    from dataset import Data
    checks=[]
    for chosen in (val[:64],val[-30:]):
        z=np.asarray(geo.npz_member_memmap(dataset,"z")[chosen])
        pos=np.asarray(geo.npz_member_memmap(dataset,"pos")[chosen])
        edge=np.asarray(geo.npz_member_memmap(dataset,"edge")[chosen])
        old=Data.__new__(Data)
        old.device="cpu";old.z=torch.from_numpy(np.array(z,copy=True))
        old.pos=torch.from_numpy(np.array(pos,copy=True))
        old.edge=torch.from_numpy(np.array(edge,copy=True))
        for k in ("E","A","f"):
            shape=(len(chosen),10,3,3) if k=="A" else (len(chosen),10)
            setattr(old,k,torch.zeros(shape,dtype=torch.float32))
        for k in ("mask_E","mask_A","mask_f"):
            setattr(old,k,torch.ones((len(chosen),10),dtype=torch.bool))
        reference,_=Data.batch(old,np.arange(len(chosen),dtype=np.int64))
        actual=fixed_input_batch(z,pos,edge,torch)
        need(set(actual)==set(reference),"adapter key mismatch")
        for k in actual:
            if k=="n":need(actual[k]==reference[k],"adapter n mismatch")
            else:need(torch.equal(actual[k],reference[k]),"adapter tensor mismatch "+k)
        checks.append({"rows":len(chosen),"exact_tensor_keys":sorted(actual)})
    return checks

def verify_hash_map(table):
    for path,digest in table.items():
        need(sha(path)==digest,"source/artifact drift "+str(path))

def check_split(train,val,test,ids):
    train=np.asarray(train,dtype=np.int64);val=np.asarray(val,dtype=np.int64)
    test=np.asarray(test,dtype=np.int64);ids=np.asarray(ids)
    need((len(train),len(val),len(test))==(120355,6686,6686),"split counts")
    allidx=np.concatenate((train,val,test))
    need(np.array_equal(np.sort(allidx),np.arange(len(ids))),"split overlap/incomplete coverage")
    need(np.unique(ids).size==len(ids),"duplicate molecule ID")
    idparts=[ids[part] for part in (train,val,test)]
    need(sum(map(len,idparts))==np.unique(np.concatenate(idparts)).size,
         "train/validation/test ID overlap")
    return {"train":len(train),"validation":len(val),"historical_test":len(test),
            "total":len(ids),"pairwise_indices_and_ids_disjoint":True,"full_coverage":True}

def validation_fixture(f):
    arrays={n:load_pred(Path(f["components"][n]["validation_reference"]["prediction_path"]),
                        f["components"][n]["validation_reference"]["prediction_sha256"]) for n in NAMES}
    a=aligned(arrays)
    need(a["indices"].shape==(6686,) and a["f_true"].shape==(6686,10),"validation population")
    preds={n:arrays[n]["f"].astype(np.float64) for n in NAMES}
    old3=sum(preds[n] for n in NAMES[:3])/3
    F5=sum(preds.values())/5
    m={k:metric(a["f_true"],v) for k,v in (("eta0",preds["mto_eta0"]),("old3",old3),("F5",F5))}
    need(abs(m["eta0"]["r2"]-0.4052941183410983)<1e-12,"eta0 validation reproduction")
    need(abs(m["old3"]["r2"]-0.4494214632744171)<1e-12,"old3 validation reproduction")
    need(abs(m["F5"]["r2"]-f["validation_r2"])<1e-12 and
         abs(m["F5"]["sse"]-f["validation_sse"])<1e-9,"F5 validation reproduction")
    return m
def ordered_members(groups):
    units=sorted(set(groups))
    values=np.asarray(groups)
    return units,[np.flatnonzero(values==u) for u in units]

def paired_bootstrap(y,reference,candidate,groups,seed=20260929,draws=2000):
    units,members=ordered_members(groups)
    rng=np.random.default_rng(seed);out=np.empty(draws,dtype=np.float64)
    for j in range(draws):
        chosen=rng.integers(0,len(units),size=len(units))
        ix=np.concatenate([members[i] for i in chosen])
        yy=y[ix];sst=np.square(yy-yy.mean(dtype=np.float64)).sum(dtype=np.float64)
        need(sst>0,"bootstrap zero SST")
        old=float(np.square(reference[ix]-yy).sum(dtype=np.float64))
        new=float(np.square(candidate[ix]-yy).sum(dtype=np.float64))
        out[j]=(old-new)/sst
    return {"units":len(units),"draws":draws,"seed":seed,
            "delta_r2_p2p5_p50_p97p5":[float(x) for x in np.quantile(out,[.025,.5,.975])],
            "positive_fraction":float((out>0).mean())}
def synthetic():
    y=np.array([[0.,1.],[2.,3.],[1.,4.]],dtype=np.float64)
    old=y+np.array([[1.,0.],[0.,1.],[1.,0.]])
    new=y+np.array([[0.,0.],[0.,1.],[0.,0.]])
    need(metric(y,new)["sse"]<metric(y,old)["sse"],"synthetic SSE")
    b=paired_bootstrap(y,old,new,["a","a","b"],draws=20)
    integer_order,integer_members=ordered_members(list(range(12)))
    need(integer_order==list(range(12)) and
         all(np.array_equal(integer_members[j],np.array([j])) for j in range(12)) and
         sorted(set(str(i) for i in range(12)))!=[str(i) for i in range(12)],
         "synthetic >=12 molecule-order regression")
    need(b["units"]==2 and b["positive_fraction"]==1.0,"synthetic group bootstrap")
    base={"ids":np.array([1,2]),"indices":np.array([3,4]),"f_true":np.ones((2,10),dtype=np.float64),
          "mask_f_true":np.ones((2,10),dtype=np.bool_),"f":np.ones((2,10),dtype=np.float64)}
    aa={n:{k:v.copy() for k,v in base.items()} for n in NAMES};aligned(aa)
    for key,value in (("ids",np.array([1,3])),("indices",np.array([3,5])),
                      ("f_true",np.zeros((2,10),dtype=np.float64)),
                      ("mask_f_true",np.zeros((2,10),dtype=np.bool_)),
                      ("f",np.full((2,10),np.nan))):
        bad={n:{k:v.copy() for k,v in base.items()} for n in NAMES}
        bad["G3"][key]=value
        try:aligned(bad)
        except RuntimeError:pass
        else:raise RuntimeError("synthetic bad "+key+" accepted")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        path=Path(td)/"validation_fixture.npz"
        np.savez(path,**base)
        try:load_pred(path,"0"*64)
        except RuntimeError:pass
        else:raise RuntimeError("synthetic changed array hash accepted")
        atomic=Path(td)/"single_use.json"
        once(atomic,{"ok":True})
        try:once(atomic,{"ok":False})
        except FileExistsError:pass
        else:raise RuntimeError("synthetic output overwrite accepted")
    fake_pins={"script_sha256":"1"*64,"freeze_sha256":"2"*64,
               "protocol_sha256":"3"*64,"preflight_sha256":"4"*64}
    fake_pre={"passed":True,**{k:v for k,v in fake_pins.items() if k!="preflight_sha256"}}
    fake_review={"passed":True,**fake_pins}
    fake_auth={"approved":True,**fake_pins,"review_sha256":"5"*64,
               "publication_commit":"6"*40,"archive_manifest_sha256":"7"*64}
    gate_records(fake_pre,fake_review,fake_auth,fake_pins,"5"*64)
    for key,bad in (("approved",False),("script_sha256","0"*64),
                    ("review_sha256","0"*64),("publication_commit",""),
                    ("archive_manifest_sha256","")):
        broken={**fake_auth,key:bad}
        try:gate_records(fake_pre,fake_review,broken,fake_pins,"5"*64)
        except RuntimeError:pass
        else:raise RuntimeError("synthetic authorization "+key+" accepted")
    with TemporaryDirectory() as td:
        a=Path(td)/"pinned.bin";a.write_bytes(b"original")
        verify_hash_map({str(a):sha(a)})
        a.write_bytes(b"changed")
        try:verify_hash_map({str(a):hashlib.sha256(b"original").hexdigest()})
        except RuntimeError:pass
        else:raise RuntimeError("changed source/checkpoint hash accepted")
    try:check_split(np.arange(120355),np.arange(120355,127041),
                    np.arange(127040,133726),np.arange(133727))
    except RuntimeError:pass
    else:raise RuntimeError("overlap/incomplete split accepted")
    with TemporaryDirectory() as td:
        place=Path(td);single_use_gate(place)
        (place/"PARTIAL.json").write_text("{}")
        try:single_use_gate(place)
        except RuntimeError:pass
        else:raise RuntimeError("preexisting partial marker accepted")
        (place/"PARTIAL.json").unlink()
        (place/"runtime").mkdir()
        (place/"runtime/G1_selected_native_test.npz").write_bytes(b"partial")
        try:single_use_gate(place)
        except RuntimeError:pass
        else:raise RuntimeError("preexisting partial prediction accepted")
    return {"group_bootstrap":True,"integer_molecule_order_ge12_checked":True,
            "changed_source_checkpoint_hash_and_incomplete_split_rejected":True,
            "preexisting_partial_marker_and_prediction_rejected":True,
            "misaligned_ids_indices_fp64_truth_mask_nonfinite_rejected":True,
            "changed_array_hash_and_output_overwrite_rejected":True,
            "missing_or_changed_authorization_rejected":True}

def single_use_gate(directory):
    for marker in ("STARTED.json","FAILED.json","INVALID.json","PARTIAL.json",
                   "TEST_RESULTS.json","TEST_REPORT.md","TEST_COMPLETE.json"):
        need(not (directory/marker).exists(),"single-use output/marker exists "+marker)
    runtime=directory/"runtime"
    for name in ("G1","G3"):
        need(not (runtime/(name+"_selected_native_test.npz")).exists(),
             "prior selected test array "+name)

def preflight():
    need(not (HERE/"PREFLIGHT.json").exists(),"preflight immutable; do not overwrite")
    f=J(FREEZE);verify_static(f)
    v=validation_fixture(f);adapter=verify_input_adapter_validation();s=synthetic()
    # The runtime gate is separately exercised with missing review/authorization;
    # preflight does not inspect any historical-test array or instantiate a model.
    for path in (HERE/"EXECUTION_AUTHORIZATION.json",HERE/"IMPLEMENTATION_REVIEW.json"):
        need(not path.exists(),"preflight unexpectedly has runtime approval "+str(path))
    result={"passed":True,"classification":"Synthetic plus saved-validation-only preparation; no test bytes/member or model forward",
            "time":time.time(),"script_sha256":sha(__file__),"freeze_sha256":sha(FREEZE),
            "protocol_sha256":sha(PROTOCOL),"validation":v,"validation_input_adapter_equivalence":adapter,"synthetic":s,
            "test_array_opened_or_hashed":False,"model_forward":False}
    once(HERE/"PREFLIGHT.json",result)
    print(json.dumps({"passed":True,"validation_F5_r2":v["F5"]["r2"]}))

def gate_records(pre,review,auth,pins,review_digest):
    need(pre.get("passed") is True and review.get("passed") is True and
         auth.get("approved") is True,"preflight/review/root authorization not PASS")
    for k,v in pins.items():
        if k!="preflight_sha256":need(pre.get(k)==v,"preflight pin "+k)
        need(review.get(k)==v,"review pin "+k)
        need(auth.get(k)==v,"authorization pin "+k)
    need(auth.get("review_sha256")==review_digest,"authorization review pin")
    commit=auth.get("publication_commit")
    manifest=auth.get("archive_manifest_sha256")
    need(isinstance(commit,str) and len(commit)==40 and
         all(c in "0123456789abcdef" for c in commit),"publication commit absent")
    need(isinstance(manifest,str) and len(manifest)==64 and
         all(c in "0123456789abcdef" for c in manifest),"D archive manifest absent")

def execution_gate(f):
    p=HERE/"PREFLIGHT.json";r=HERE/"IMPLEMENTATION_REVIEW.json";a=HERE/"EXECUTION_AUTHORIZATION.json"
    need(p.is_file() and r.is_file() and a.is_file(),"missing preflight/review/root authorization")
    pre,review,auth=J(p),J(r),J(a)
    pins={"script_sha256":sha(__file__),"freeze_sha256":sha(FREEZE),
          "protocol_sha256":sha(PROTOCOL),"preflight_sha256":sha(p)}
    gate_records(pre,review,auth,pins,sha(r))
    return {"preflight_sha256":sha(p),"review_sha256":sha(r),"authorization_sha256":sha(a),
            "publication_commit":auth["publication_commit"],"archive_manifest_sha256":auth["archive_manifest_sha256"]}

def score_slices(y,p,pooled_mean,pooled_sst,thresholds,q2bins):
    overall=metric(y,p)
    states=[]
    for j in range(10):
        m=metric(y[:,j],p[:,j])
        m["state"]=j+1
        m["pooled_sst_contribution"]=float(np.square(y[:,j]-pooled_mean).sum(dtype=np.float64)/pooled_sst)
        states.append(m)
    tails={}
    for label,t in thresholds.items():
        bright=y>=t
        entry={}
        for mode,mask in (("bright",bright),("below",~bright)):
            err=p[mask]-y[mask]
            entry[mode]={"count":int(mask.sum()),"sse":float(np.square(err).sum(dtype=np.float64)),
                         "mae":float(np.abs(err).mean()) if err.size else None}
        tails[label]={"threshold":t,**entry}
    bins=[]
    for j in [-1,*range(int(q2bins.max())+1)]:
        rows=q2bins==j
        err=p[rows]-y[rows]
        bins.append({"q2_bin":j,"molecules":int(rows.sum()),"labels":int(err.size),
                     "sse":float(np.square(err).sum(dtype=np.float64)),
                     "share_of_total_sse":float(np.square(err).sum(dtype=np.float64)/overall["sse"])})
    molerr=np.square(p-y).sum(axis=1,dtype=np.float64)
    top=np.sort(molerr)[::-1]
    concentration={}
    for label,n in (("top1",1),("top10",10),("top1pct",int(np.ceil(.01*len(top))))):
        sse=float(top[:n].sum(dtype=np.float64))
        concentration[label]={"molecules":n,"sse":sse,"share_of_total_sse":sse/overall["sse"]}
    overall["negative_prediction_count"]=int((p<0).sum())
    overall["nonfinite_prediction_count"]=int((~np.isfinite(p)).sum())
    return {"overall":overall,"states":states,"TRAIN_tail_slices":tails,
            "TRAIN_q2_bins":bins,"molecule_error_concentration":concentration}

def error_change(y,reference,candidate,ids,groups):
    gain=np.square(reference-y).sum(axis=1,dtype=np.float64)-np.square(candidate-y).sum(axis=1,dtype=np.float64)
    net=float(gain.sum(dtype=np.float64))
    positive=float(gain[gain>0].sum(dtype=np.float64));negative=float(gain[gain<0].sum(dtype=np.float64))
    order_pos=np.argsort(-gain,kind="stable");order_neg=np.argsort(gain,kind="stable")
    def rows(ix):
        return [{"id":int(ids[i]),"reference_sse":float(np.square(reference[i]-y[i]).sum()),
                 "candidate_sse":float(np.square(candidate[i]-y[i]).sum()),
                 "sse_gain":float(gain[i])} for i in ix]
    keys=sorted(set(groups));group_gain=[]
    gg=np.asarray(groups)
    for k in keys:group_gain.append(float(gain[gg==k].sum(dtype=np.float64)))
    group_gain=np.asarray(group_gain)
    def group_row(j):
        key=keys[int(j)];mask=gg==key
        return {"connectivity_key_sha256":hashlib.sha256(str(key).encode()).hexdigest(),
                "molecule_count":int(mask.sum()),"molecule_ids":[int(i) for i in ids[mask]],
                "reference_sse":float(np.square(reference[mask]-y[mask]).sum(dtype=np.float64)),
                "candidate_sse":float(np.square(candidate[mask]-y[mask]).sum(dtype=np.float64)),
                "sse_gain":float(group_gain[int(j)])}
    order_gp=np.argsort(-group_gain,kind="stable")
    order_gn=np.argsort(group_gain,kind="stable")
    return {"net_sse_gain_reference_minus_candidate":net,"gross_molecule_gain":positive,
            "gross_molecule_loss":negative,
            "gross_group_gain":float(group_gain[group_gain>0].sum(dtype=np.float64)),
            "gross_group_loss":float(group_gain[group_gain<0].sum(dtype=np.float64)),
            "molecules_improved":int((gain>0).sum()),
            "molecules_worsened":int((gain<0).sum()),"groups_improved":int((group_gain>0).sum()),
            "groups_worsened":int((group_gain<0).sum()),
            "largest_positive_molecule":rows(order_pos[:1]) if (gain>0).any() else None,
            "largest_negative_molecule":rows(order_neg[:1]) if (gain<0).any() else None,
            "top10_positive_molecules":rows([i for i in order_pos if gain[i]>0][:10]),
            "top10_negative_molecules":rows([i for i in order_neg if gain[i]<0][:10]),
            "largest_positive_group":group_row(order_gp[0]) if (group_gain>0).any() else None,
            "largest_negative_group":group_row(order_gn[0]) if (group_gain<0).any() else None,
            "top10_positive_groups":[group_row(j) for j in order_gp if group_gain[j]>0][:10],
            "top10_negative_groups":[group_row(j) for j in order_gn if group_gain[j]<0][:10],
            "positive_gain_share_headline":None if net<=0 else positive/net}

def evaluate():
    # No test member or old test prediction is opened until this gate, health
    # admission, and the durable single-use STARTED receipt all succeed.
    f=J(FREEZE);verify_static(f);gate=execution_gate(f)
    single_use_gate(HERE)
    runtime=HERE/"runtime";runtime.mkdir(exist_ok=True)
    need(os.getenv("CUDA_VISIBLE_DEVICES") in (None,""),"physical CUDA index remapped")
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    need(sha(ROOT/"architecture/gpu_health.py")==GPU_HEALTH_SHA,"GPU health source changed")
    sys.path.insert(0,str(ROOT/"architecture"))
    from gpu_health import snapshot,eligible,unchanged
    lock=Path("/tmp/mto_pouter_gpu_4.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=snapshot(4)
    need(eligible(before) and not before["apps"] and before["uuid"]==GPU_UUID,"GPU4 admission failed")
    started={"event":"TEST_STARTED","time":time.time(),"pid":os.getpid(),
             "process_start_ticks":Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
             "cwd":str(Path.cwd()),"gpu_before":before,**gate,
             "freeze_sha256":sha(FREEZE),"protocol_sha256":sha(PROTOCOL),"script_sha256":sha(__file__)}
    once(HERE/"STARTED.json",started)
    try:
        run_test(f,before,snapshot,unchanged,torch)
    except BaseException as e:
        fail={"event":"TEST_FAILED_OR_PARTIAL","time":time.time(),"error_type":type(e).__name__,
              "error":str(e),"started_sha256":sha(HERE/"STARTED.json"),
              "existing_runtime_files":[str(x) for x in sorted(runtime.iterdir())]}
        once(HERE/"FAILED.json",fail)
        raise

def run_test(f,before,snapshot,unchanged,torch):
    src=BASE/"experiments/qm9s_chan64_20260928"
    sys.path.insert(0,str(src))
    from model_factory import build
    geo=pinned_geometry()
    dataset=src/"data/dataset.npz";rawpath=src/"data/raw_labels.npz"
    ids_all=geo.npz_member_memmap(dataset,"ids")
    train=np.asarray(geo.npz_member_memmap(dataset,"train"),dtype=np.int64)
    val=np.asarray(geo.npz_member_memmap(dataset,"val"),dtype=np.int64)
    idx=np.asarray(geo.npz_member_memmap(dataset,"test"),dtype=np.int64)
    split_integrity=check_split(train,val,idx,ids_all)
    raw_ids=geo.npz_member_memmap(rawpath,"ids")
    need(np.array_equal(np.asarray(raw_ids[idx]),np.asarray(ids_all[idx])),
         "fixed test raw ID mismatch")
    y=np.asarray(geo.npz_member_memmap(rawpath,"f")[idx],dtype=np.float64)
    m_f=np.asarray(geo.npz_member_memmap(rawpath,"mask_f")[idx],dtype=np.bool_)
    m_e=np.asarray(geo.npz_member_memmap(rawpath,"mask_E")[idx],dtype=np.bool_)
    m_a=np.asarray(geo.npz_member_memmap(rawpath,"mask_A")[idx],dtype=np.bool_)
    mask=m_f&m_e&m_a
    energy_true=np.asarray(geo.npz_member_memmap(rawpath,"E")[idx],dtype=np.float64)
    ids=np.asarray(ids_all[idx],dtype=np.int64)
    stats=J(src/"data/normalization.json")
    weight=J(src/"data/trace_weight.json")
    need(float(weight["mean_E2_train"])>0,"invalid frozen trace weight")
    stats["mean_E2_train"]=float(weight["mean_E2_train"])
    need(y.shape==(6686,10) and mask.all() and np.isfinite(y).all(),"raw test truth/mask")
    val_ids=np.asarray(ids_all[val],dtype=np.int64)
    need(np.intersect1d(ids,val_ids).size==0,"validation/test ID overlap")
    arrays={}
    for name in NAMES[:3]:
        c=f["components"][name];meta=c["historical_test_array"]
        arrays[name]=load_pred(Path(meta["path"]),meta["sha256_from_historical_receipt"])
        a=arrays[name]
        need(np.array_equal(a["indices"],idx) and np.array_equal(a["ids"],ids) and
             np.array_equal(a["f_true"],y) and np.array_equal(a["mask_f_true"],mask),
             "historical eta test alignment "+name)
    old=aligned(arrays,NAMES[:3])
    old3=sum(arrays[n]["f"].astype(np.float64) for n in NAMES[:3])/3
    old_eta=arrays["mto_eta0"]["f"].astype(np.float64)
    historical=J(ROOT/"ensemble_diagnostic/EXPLORATORY_TEST.json")
    need(abs(metric(y,old3)["r2"]-historical["ensemble"]["r2"])<=1e-10 and
         abs(metric(y,old_eta)["r2"]-historical["baseline"]["r2"])<=1e-10,
         "old3/eta0 historical test reproduction")
    device=torch.device("cuda:4");new_receipts={}
    for name in ("G1","G3"):
        c=f["components"][name]
        config=c["config"];ck=torch.load(c["selected_checkpoint_path"],map_location="cpu",weights_only=False)
        need(ck["config"]==config and ck["epoch"]==c["epoch"],"selected checkpoint metadata "+name)
        model=build(config,stats).to(device);model.load_state_dict(ck["model"]);model.eval()
        eparts=[];tparts=[];toldparts=[];begin=time.monotonic()
        with torch.no_grad():
            for a in range(0,len(idx),64):
                chosen=idx[a:a+64]
                zrow=np.asarray(geo.npz_member_memmap(dataset,"z")[chosen])
                prow=np.asarray(geo.npz_member_memmap(dataset,"pos")[chosen])
                erow=np.asarray(geo.npz_member_memmap(dataset,"edge")[chosen])
                x=fixed_input_batch(zrow,prow,erow,torch)
                x={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in x.items()}
                E,A=model(**x)
                eparts.append(E.cpu().numpy())
                tparts.append(A.double().diagonal(dim1=-2,dim2=-1).sum(-1).cpu().numpy())
                toldparts.append(A.diagonal(dim1=-2,dim2=-1).sum(-1).cpu().numpy())
        elapsed=time.monotonic()-begin
        en=np.concatenate(eparts).astype(np.float64)
        trace=np.concatenate(tparts).astype(np.float64)
        trace_old=np.concatenate(toldparts).astype(np.float64)
        native=K*en*trace
        need(native.shape==y.shape and np.isfinite(native).all() and np.isfinite(en).all(),
             "nonfinite native output "+name)
        path=HERE/"runtime"/(name+"_selected_native_test.npz")
        need(not path.exists(),"existing selected test artifact "+name)
        temp=path.with_suffix(".tmp.npz")
        np.savez_compressed(temp,ids=ids,indices=idx,f=native,f_true=y,
                            mask_f_true=mask,E_pred=en,E_true=energy_true,trace_pred=trace,
                            checkpoint_sha256=np.asarray(c["checkpoint_sha256"]),
                            run_manifest_sha256=np.asarray(c["run_manifest_sha256"]))
        os.replace(temp,path)
        arrays[name]={"ids":ids.copy(),"indices":idx.copy(),"f":native,"f_true":y.copy(),
                      "mask_f_true":mask.copy()}
        new_receipts[name]={"checkpoint_sha256":c["checkpoint_sha256"],"selected_epoch":c["epoch"],
                            "prediction_sha256":sha(path),"inference_seconds":elapsed,
                            "checkpoint_bytes":Path(c["selected_checkpoint_path"]).stat().st_size,
                            "negative_prediction_count":int((native<0).sum()),
                            "legacy_FP32_trace_metric":metric(y,K*en*trace_old)}
        del model,ck
        torch.cuda.empty_cache()
    aligned(arrays)
    F5=sum(arrays[n]["f"].astype(np.float64) for n in NAMES)/5
    overall_mean=float(y.mean(dtype=np.float64))
    pooled_sst=float(np.square(y-overall_mean).sum(dtype=np.float64))
    # Consume only fixed historical-test rows after authorization.
    z=np.asarray(geo.npz_member_memmap(dataset,"z")[idx])
    pos=np.asarray(geo.npz_member_memmap(dataset,"pos")[idx])
    q2=np.asarray([geo.geometry(zz,pp)[0] for zz,pp in zip(z,pos)],dtype=np.float64)
    bounds=np.asarray(f["prospective_diagnostics"]["q2_boundaries"],dtype=np.float64)
    bins=np.where(np.isfinite(q2),np.searchsorted(bounds,q2,side="left"),-1)
    identity_path=src/"data/identity_audit_v2.json"
    identity,_=geo.stream_selected_identity_rows(identity_path,idx)
    need(all(int(identity[int(i)][0])==int(mid) for i,mid in zip(idx,ids)),"identity/ID mismatch")
    groups=[str(identity[int(i)][1]) for i in idx]
    preds={"eta0":old_eta,"old3":old3,"F5":F5}
    thresholds=f["prospective_diagnostics"]["raw_f_tail_thresholds"]
    summaries={n:score_slices(y,p,overall_mean,pooled_sst,thresholds,bins) for n,p in preds.items()}
    bootstrap={"molecule":paired_bootstrap(y,old3,F5,list(range(len(y)))),
               "connectivity_group":paired_bootstrap(y,old3,F5,groups)}
    delta=summaries["F5"]["overall"]["r2"]-summaries["old3"]["overall"]["r2"]
    change=error_change(y,old3,F5,ids,groups)
    after=snapshot(4);need(unchanged(before,after),"GPU4 health changed")
    result={"classification":"One frozen F5 versus old3 exploratory historical-test comparison; test historically reused",
            "time":time.time(),"freeze_sha256":sha(FREEZE),"protocol_sha256":sha(PROTOCOL),
            "script_sha256":sha(__file__),"started_sha256":sha(HERE/"STARTED.json"),
            "historical_eta_test_array_hashes":{n:f["components"][n]["historical_test_array"]["sha256_from_historical_receipt"] for n in NAMES[:3]},
            "new_selected_inference":new_receipts,"validation_test_indices_ids_disjoint":True,
            "full_population":{"molecules":len(y),"labels":int(y.size),"all_masks_valid":True},"split_integrity":split_integrity,
            "predictors":summaries,"primary_F5_minus_old3_delta_r2":delta,
            "primary_relative_sse_reduction":(summaries["old3"]["overall"]["sse"]-
                summaries["F5"]["overall"]["sse"])/summaries["old3"]["overall"]["sse"],
            "paired_bootstrap":bootstrap,"primary_error_change":change,
            "TRAIN_tail_thresholds":thresholds,"TRAIN_q2_boundaries":bounds.tolist(),
            "gpu_before":before,"gpu_after":after,"test_reused_historical":True,
            "selection_changed_after_test":False,"extra_model_forwards":2,
            "deployment_model_forwards":{"F5":5,"old3":3,"eta0":1}}
    once(HERE/"TEST_RESULTS.json",result)
    lines=["# Frozen F5 historical-test comparison","","The historical test has been reused in earlier campaigns; this is exploratory and not an untouched holdout.","",
           "| Predictor | Pooled raw-f SSE | R² | RMSE | MAE |","|---|---:|---:|---:|---:|"]
    for name in ("eta0","old3","F5"):
        v=summaries[name]["overall"]
        lines.append(f"| {name} | {v['sse']:.9f} | {v['r2']:.9f} | {v['rmse']:.8f} | {v['mae']:.8f} |")
    lines += ["",f"Primary F5−old3 ΔR²: {delta:+.9f}; relative SSE reduction {result['primary_relative_sse_reduction']:+.6%}.",
              f"Paired molecule bootstrap: {bootstrap['molecule']['delta_r2_p2p5_p50_p97p5']}.",
              f"Paired connectivity bootstrap: {bootstrap['connectivity_group']['delta_r2_p2p5_p50_p97p5']}.",
              "All states, TRAIN-derived tail and q2 slices, signed errors, and contribution/concentration diagnostics are in JSON.",
              "Five model forwards are needed for deployment versus three for old3. Only two new selected chan64 forwards were run here.",
              "No new training, tuning, clipping, model selection or test-driven change was made.",""]
    report=HERE/"TEST_REPORT.md"
    fd=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"w",encoding="utf-8") as out:out.write("\n".join(lines))
    complete={"event":"TEST_COMPLETE","time":time.time(),"started_sha256":sha(HERE/"STARTED.json"),
              "result_sha256":sha(HERE/"TEST_RESULTS.json"),"report_sha256":sha(report),
              "runtime_prediction_hashes":{n:r["prediction_sha256"] for n,r in new_receipts.items()},
              "gpu_after":after}
    once(HERE/"TEST_COMPLETE.json",complete)
    print(json.dumps({"F5_r2":summaries["F5"]["overall"]["r2"],
                      "old3_r2":summaries["old3"]["overall"]["r2"],"delta_r2":delta}))

if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight",action="store_true");g.add_argument("--evaluate",action="store_true")
    args=ap.parse_args()
    if args.preflight:preflight()
    else:evaluate()
