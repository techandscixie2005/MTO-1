# Affine-stage resource placement observation

The fixed33 source completed normally on its registered GPU1 process. The terminal checkpoint/hash/process gate passed. This note concerns the subsequent short affine-fit process, not source fitting.

The first affine command was the reviewed pinned Python invocation `affine_stage.py fit --gpu 1`, without an external CUDA mask. The frozen script admitted physical GPU1 and took lock1, but assigned CUDA_VISIBLE_DEVICES inside Python after imports. The command exited0 and wrote immutable coefficients and both composite exports. Its log is AFFINE_FIT_EXECUTION_20260929T2001252036053Z.log. Those outputs are retained; the fit is not repeated.

Monitor record `monitoring/ROUND03_HEARTBEAT_20260929T195916.json` observed an MTO-environment Python PID1186535 with504MiB on GPU0 alongside the preserved SpecGPT process. At that snapshot GPU1 had only Xorg and13MiB total use. The MTO PID had exited before argv/start identity could be captured. Its timing and executable are consistent with our affine invocation, but the observation does not positively bind its argv or show the location of every inference operation. We cannot claim the fit ran entirely on admitted GPU1 or that GPU0 was never used. The exact import/context cause has not been demonstrated. No process was signaled and no unrelated job was modified; performance impact was not measured.

Root accepted an execution-environment correction preserving all scientific sources and frozen coefficients. History reviewed the exact command/mapping and stdlib launcher. Validation receives this environment BEFORE its interpreter starts:

```
CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 PYTHONUNBUFFERED=1 \
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py evaluate --gpu 1
```

Physical GPU1 is therefore the intended sole visible logical CUDA device0. The unchanged child still performs physical GPU1 health admission and shared locking. `run_validation_bound.py` (reviewed SHA810c6349b7f87e59e4711d84ed088281d58e382c8574583ff587487af64d7645) records the exact command/cwd/environment, owned PID/start identity, filtered own-PID device observations, exit status and unchanged coefficient/export hashes. The final VALIDATION_LAUNCH_<attempt>.json is authoritative for whether actual GPU1 placement was observed; absence of a contradictory observation alone is insufficient proof.

No new source fit, coefficient fit, hyperparameter choice or completed-validation repeat is authorized by this correction. Future reproduction should prebind the physical GPU UUID before importing CUDA libraries. Preserve this resource-contract limitation separately from the numerical and scientific audit.
