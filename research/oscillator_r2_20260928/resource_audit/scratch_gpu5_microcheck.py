#!/usr/bin/env python3
"""Fresh GPU5 admission for scratch study: synthetic matrix and fixed TRAIN model probes only."""
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison")
ARCH=ROOT.parent/"architecture"
sys.path.insert(0,str(ARCH));sys.path.insert(0,str(ROOT))
from common import source_seal,code_hashes,sha,save_json,stats,setup,TrainValData,objective
from model import NativeDirect,MTODirect
import gpu_health
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="5" and os.environ.get("MTO_PHYSICAL_GPU")=="5"
    lock=open("/tmp/mto_pouter_gpu_5.lock","a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=gpu_health.snapshot(5);assert gpu_health.eligible(before,guarded=True),before
    source=source_seal();code=code_hashes();setup(11)
    torch.manual_seed(20260929);a=torch.randn(256,256,dtype=torch.float64)
    b=torch.randn(256,256,dtype=torch.float64)
    expected64=a@b
    got64=a.cuda()@b.cuda();got64repeat=a.cuda()@b.cuda()
    max64=float((got64.cpu()-expected64).abs().max())
    repeat64=float((got64-got64repeat).abs().max())
    assert torch.allclose(got64.cpu(),expected64,rtol=1e-10,atol=1e-9),(max64,repeat64)
    assert repeat64<=1e-12
    af=a.float();bf=b.float();cpu32=af@bf;gpu32=af.cuda()@bf.cuda()
    max32=float((gpu32.cpu()-cpu32).abs().max())
    assert torch.allclose(gpu32.cpu(),cpu32,rtol=1e-4,atol=2e-4),max32
    data=TrainValData("cpu");ix=data.parts["train"][:16];xc,yc=data.batch(ix)
    xg={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in xc.items()}
    yg={k:v.cuda() for k,v in yc.items()}
    result={}
    for arm,cls in (("native",NativeDirect),("mto",MTODirect)):
        state=torch.load(ROOT/"initial"/f"{arm}.pt",map_location="cpu",weights_only=False)["model"]
        cpu=cls(stats());cpu.load_state_dict(state)
        gpu=cls(stats()).cuda();gpu.load_state_dict(state)
        # Frozen nonzero terminal readout probe, so output comparison tests the backbone.
        with torch.no_grad():
            for m in (cpu,gpu):
                if arm=="native":
                    final=[q for q in m.base.sout.modules() if isinstance(q,torch.nn.Linear)][-1]
                    final.weight.fill_(.03/128);final.bias.zero_()
                else:
                    m.base.decoder.energy_head.weight.fill_(.03/128)
                    m.base.decoder.energy_head.bias.zero_()
                    m.strength_head[-1].weight.fill_(.03/32)
                    m.strength_head[-1].bias.zero_()
        cpu.eval();gpu.eval()
        with torch.no_grad():
            pc=cpu(**xc);pg=gpu(**xg);pg2=gpu(**xg)
        row={}
        for key in ("E","f"):
            diff=float((pg[key].cpu()-pc[key]).abs().max())
            repeat=float((pg[key]-pg2[key]).abs().max())
            span=float((pg[key][0]-pg[key][-1]).abs().max())
            assert torch.isfinite(pg[key]).all() and (pg[key]>0).all() and span>1e-7
            assert torch.allclose(pg[key].cpu(),pc[key],rtol=2e-4,atol=2e-5),(arm,key,diff)
            assert repeat<=2e-5,(arm,key,repeat)
            row[key]={"cpu_gpu_max_abs":diff,"gpu_repeat_max_abs":repeat,"molecule_span":span}
        gpu.train();gpu.zero_grad(set_to_none=True)
        v=objective(gpu(**xg),yg,stats());v[0].backward()
        grads=[p.grad for p in gpu.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        row["loss"]=float(v[0]);row["max_grad"]=max(float(g.abs().max()) for g in grads)
        result[arm]=row
    torch.cuda.synchronize()
    after=gpu_health.snapshot(5)
    assert gpu_health.unchanged(before,after) and set(after["apps"]).issubset({os.getpid()})
    assert before["ecc"]["aggregate"]["dram_correctable"]==1
    receipt={"passed":True,"microcheck_passed":True,"classification":"New GPU5 synthetic+fixed TRAIN numerical admission; no training/validation/test inference",
        "gpu_uuid":before["uuid"],"snapshot":before,"after":after,
        "source_hashes":source,"code_hashes":code,
        "microcheck_script_sha256":sha(__file__),
        "fp64_gemm_max_abs":max64,"fp64_repeat_max_abs":repeat64,
        "fp32_gemm_max_abs":max32,"model_checks":result,
        "historical_corrected_dram_count":1,"test_batches":0,"val_inference_batches":0,
        "train_molecules_inferred":16,"time":time.time()}
    save_json(receipt,ROOT/"GPU5_ADMISSION.json")
    print(json.dumps({"passed":True,"gpu":5,"gemm64":max64,"gemm32":max32,
        "model":result}),flush=True)
if __name__=="__main__":main()

