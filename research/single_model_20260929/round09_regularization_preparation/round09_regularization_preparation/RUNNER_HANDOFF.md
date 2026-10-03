# Round09 future production and recovery handoff

**Production is blocked.** Preparation PASS, acceptance, archive or publication alone grant no execution authority. A later explicit instruction must permit fitting before root issues the exact bound authorization below. Otherwise future heartbeats remain monitoring/preparation-only. Completed numerical preparation stages must not repeat.

Namespace: /home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation
Pinned Python: /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python

## Exact authority and command

Root's PRODUCTION_EXECUTION_AUTHORIZATION.json must match the template: authorized=true; scope round09_two_arm_60epoch_fit; ordered arms [zero_decay,coupled_l2]; epochs60; test_access=false; historical_weights=false. Bind exact FROZEN_MANIFEST.json, INDEPENDENT_PREPARATION_REVIEW.json and campaign ops/ROUND09_PREPARATION_PUBLICATION_RECEIPT.json SHA values, plus splitc8ce66dd/verifier395d415f. Review must bind the entire source dictionary; publication must have remote_verified and download_before_stage_before_commit_push true and the identical source/review hashes. The runner rehashes every file before any scientific import or data access. Current false template fails closed.

Only after that separate authority:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
 launch.py --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms zero_decay coupled_l2
```

Admissions are sequential; after admission the two workers may train concurrently on separate GPUs. zero_decay uses physicalGPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800; coupled_l2 uses physicalGPU2 UUID GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa. Each requires immediate idle/ECC/remap/recovery health checks and exclusive /tmp/mto_pouter_gpu_<index>.lock. UUID visibility is set before interpreter imports; logicalcuda:0 is the single admitted device. No fallback; GPU0 and unrelated jobs remain untouched. An admission failure on a later arm does not cancel an earlier healthy registered arm; report exact partial state for root decision.

Child OS argv first executes registered_entry.py then train.py. The same PID waits on the inherited pipe until registration succeeds for the exact PID/authority SHA. Failed/timed-out registration closes the barrier, stops only the owned child and preserves evidence; no scientific imports/updates or automatic retry. Shared GPU and worker locks remain inherited/held.

## Runtime and recovery

Each runs/<arm>/attempts/<unique-id> preserves admission XML, full stable identity, command/environment, registration and log. Arm status/history/BEST/last.pt/best.pt/FIT_COMPLETE describe progress. Both arms retain the same frozen F/transport schema and exact135 trainables/1,552,092 elements. Weight decay is the sole optimizer difference. Verify immutable PARAMETER_ROSTER SHA290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943 before update/resume. All model/optimizer/prediction data remain server-only.

last.pt commits a completed epoch with model/optimizer/AMSGrad/RNG/order/history/best/source/split/roster/decay. Resume it, never the selected best snapshot. Replay an interrupted uncommitted epoch with fixed order. CUDA equality is a fixed numerical preflight contract, not whole-trajectory determinism. Disposable preflight states cannot initialize either run; production reconstructs fresh seed11 and verifies exact base/full hashes.

An application interruption is not evidence of worker failure. Inspect stable registered PID/start/boot/UID/cwd/argv, status and terminal markers first. Preserve genuine failure/logs and do not signal unknown jobs or remove artifacts. Only an explicit reviewed recovery can invoke the same launcher for an affected arm, using unchanged source/settings/authority and a new attempt/registry ID. It reacquires health/locks and resumes committed last.pt. FIT_COMPLETE refuses rerun. No changed decay/groups/clip/LR, extension or TEST access.

Normal completion requires60epochs/112860steps,60 fixed order hashes, unchanged sources/roster/split, opaque checkpoint/export hashes and zeroTEST numeric reads. Analyze saved selected/fixed60 validation predictions once under separate reviewed closeout code. The coupled_l2 allocation gate remains +.003 over contemporary zero_decay AND retained.44716940136585204. No automatic seed allocation. Best-model export is one geometry checkpoint with strict config/stats/mode/contract; command imports predictor.load_predictor and calls (z,pos,batch,n,edge_index), with no old weights or training data dependency.

## Current preparation stages

CPU stage completed once in ops/CPU_EXECUTION_RECEIPT.json: five toy steps, zero model updates, roster frozen before all steps. Do not repeat. The original ops/gpu_preflight_attempt is an intentionally preserved empty resource-rejection attempt with zero data/updates. GPU fixture completion must be read from the distinct ops/gpu_preflight_attempt_retry01/COMPLETE.json plus GPU_PREFLIGHT.json and independent review. A missing stage is not permission to bypass its source/admission gate. No production authority is present during preparation. Final preparation supplement uses INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json; the original proposal INDEPENDENT_REVIEW_MANIFEST.json remains sealed unchanged.

