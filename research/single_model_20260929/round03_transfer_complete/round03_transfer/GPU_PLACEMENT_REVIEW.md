# Round03 affine evaluation device binding review

The fixed33 source completed and monitor independently verified its original PID1174422 was absent, with matching final and resumable checkpoint hashes. The later affine-fit process exited successfully and produced the frozen coefficient/export receipts. No fit is repeated.

## Completed-fit limitation

Monitor's `monitoring/ROUND03_HEARTBEAT_20260929T195916.json` observed Python PID1186535 using504MiB on physicalGPU0 while GPU1 was effectively idle. Science identifies this PID as its affine-fit child, but contemporaneous argv, start identity and working directory were not captured before exit. The memory observation does not establish the location of all model inference or justify calling the context incidental. Exact completed-fit GPU placement remains unverified. Its frozen coefficients and exports are retained, with this provenance limitation.

## Pending evaluation command

Root authorized binding CUDA visibility before interpreter startup, without changing the frozen scientific program, coefficients or settings:

```sh
CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 PYTHONUNBUFFERED=1 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py evaluate --gpu 1
```

Run from `/home/inspur/MTO-1/research/single_model_20260929/round03_transfer`. If a stdlib launcher is used, it must pass this UUID in the child environment before `Popen` and retain the same program arguments and working directory. Record child PID, start identity, inherited binding and observed GPU UUID. Existing admission and locking remain authoritative; a busy or unhealthy device still causes refusal.

Source review confirms the physical/logical distinction: `admit(1)` invokes `nvidia-smi -i 1` and the program locks `/tmp/mto_pouter_gpu_1.lock`. With one UUID visible from process startup, every bare `.cuda()` and `device='cuda'` target is logical0, mapped to physicalGPU1. The program's later environment assignment uses the same UUID. Prebinding also covers any imported-library CUDA initialization before `main`.

Reviewed `affine_stage.py`, `runtime.py`, `clean_source.py`, the frozen MTO/DetaNet imports, and parent `launch.py`/`freeze.py`. This is a placement review of the exact pending command, not a completion or metric claim. Science remains the sole evaluation executor. The reviewer performs no model inference, fitting, coefficient changes or frozen source changes. Native and historical-affine validation replay anchors must still pass.

Verdict: **PASS for the prebound evaluation invocation**, subject to its existing source/hash/resource gates. Retain the completed-fit placement caveat in round closeout.

## Bound launcher review

Reviewed `run_validation_bound.py`, SHA256 `810c6349b7f87e59e4711d84ed088281d58e382c8574583ff587487af64d7645`: **PASS**. The stdlib-only launcher sets the UUID in the child environment before `Popen`, preserves the exact approved command/cwd, verifies pinned files before and after, and records child identity, filtered own-PID GPU observations and actual exit code. Final review will distinguish positive placement observations from an empty observation list. No completed evaluation is claimed by this source approval.
