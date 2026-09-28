#!/usr/bin/env python3
"""One-time frozen eta0 train/validation feature cache. No test partition is loaded."""
import fcntl,json,os,sys,time
from pathlib import Path
from common import ROOT,ARCH,RUNTIME,SRC,sha,atomic_json,atomic_npz,config,pinned_sources,Split,setup_torch,source_model,feature_cache
def main():
    import torch
    cfg=config();gpu=int(os.environ["MTO_PHYSICAL_GPU"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==str(gpu)
    assert gpu in cfg["allowed_physical_gpus"] and torch.cuda.device_count()==1
    assert not (ROOT/"CACHE_MANIFEST.json").exists(),"Cache already prepared"
    assert not (RUNTIME/"cache_train.npz").exists() and not (RUNTIME/"cache_val.npz").exists()
    lock=open(f"/tmp/mto_pouter_gpu_{gpu}.lock","a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ARCH))
    from gpu_health import snapshot,eligible,unchanged
    baseline=snapshot(gpu);assert eligible(baseline,guarded=gpu==5)
    if gpu==5:
        raise RuntimeError("Initial cache admission uses clean GPU1/4/6; guarded GPU5 requires separate microcheck")
    source=pinned_sources()
    setup_torch(cfg["seed"])
    model=source_model("cuda")
    assert sum(p.numel() for p in model.base.parameters())==1552092
    for p in model.parameters():p.requires_grad_(False)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    sections={}
    def guard():
        now=snapshot(gpu)
        if not unchanged(baseline,now) or not set(now["apps"]).issubset({os.getpid()}):
            raise RuntimeError("GPU health changed or foreign process appeared")
    for part in ("train","val"):
        guard();split=Split(part)
        arrays,seconds=feature_cache(split,model,"cuda",cfg["feature_batch_size_molecules"],guard)
        guard()
        path=RUNTIME/(f"cache_{part}.npz")
        atomic_npz(arrays,path)
        sections[part]={"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size,
            "shape_h":list(arrays["h"].shape),"dtype_h":str(arrays["h"].dtype),
            "shape_scalar":list(arrays["E_pred"].shape),"dtypes":{k:str(v.dtype) for k,v in arrays.items()},
            "molecules":len(split.ids),"first_global_index":int(split.indices[0]),
            "first_id":int(split.ids[0]),"generation_seconds":seconds}
        del arrays,split
    final=snapshot(gpu);assert unchanged(baseline,final)
    assert set(final["apps"]).issubset({os.getpid()})
    manifest={"classification":"Frozen eta0 train/validation features, server-only runtime cache",
        "created":time.time(),"physical_gpu":gpu,"health_before":baseline,"health_after":final,
        "source_hashes":source,"cache_script_sha256":sha(Path(__file__)),
        "model_eval_mode":True,"torch_tf32_disabled":True,"train_test_loader_constructed":False,
        "cache":sections,"initial_head_checkpoint_sha256":sha(ARCH/"initial/retained_residual.pt")}
    atomic_json(manifest,ROOT/"CACHE_MANIFEST.json")
    print(json.dumps({"complete":True,"gpu":gpu,"cache":{k:{"bytes":v["bytes"],
        "seconds":v["generation_seconds"],"sha256":v["sha256"]} for k,v in sections.items()}}))
if __name__=="__main__":main()

