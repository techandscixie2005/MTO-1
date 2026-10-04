# Round10 production and recovery contract

Production remains blocked. Preparation review/acceptance/publication alone grant no fitting rights. No completed technical stage may repeat. Preserve PROTOCOL_PROPOSAL.md and its15-file INDEPENDENT_REVIEW_MANIFEST.json; final preparation supplement must use a distinct name.

Namespace: /home/inspur/MTO-1/research/single_model_20260929/round10_schedule_preparation

## Future authority and command

Only after later root fitting authority, root writes PRODUCTION_EXECUTION_AUTHORIZATION.json with authorized=true, scope round10_two_arm_60epoch_fit, ordered arms[fixed_lr,step_lr], epochs60, test_access=false, historical_weights=false. Bind exact FROZEN_MANIFEST, INDEPENDENT_PREPARATION_REVIEW and verified campaign ops/ROUND10_PREPARATION_PUBLICATION_RECEIPT hashes, splitc8ce66dd and verifier395d415f. Review must bind the full source dictionary; publication must assert remote_verified and download_before_stage_before_commit_push with identical source/review. All source hashes are checked before scientific imports/data/run artifacts. The current false template rejects.

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round10_schedule_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
 launch.py --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms fixed_lr step_lr
```

Production admissions are sequential: fixed_lr physicalGPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800; step_lr physicalGPU2 UUID GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa. Once independently admitted, workers may run concurrently on those separate devices. Strict fresh health/occupancy checks, exclusive shared GPU locks and UUID visibility before interpreter imports are required. Logicalcuda0 is the single admitted UUID. No production fallback is granted by technical GPU1/2/4/6 selection. Report any rejected or partial launch for a concrete root decision, preserving earlier healthy workers and unrelated jobs.

The child OS argv begins registered_entry.py then train.py. That PID waits on the pipe for exact registered PID/authority binding before scientific imports. Failure/timeout closes the barrier, stops only the owned child and preserves evidence. No automatic retry. Worker and shared GPU locks remain held/inherited.

## Schedule and recovery

Both arms have the same fresh seed11 tensors, ordered135 trainable parameters and original LE+Ls/WD0 AdamAMSGrad. fixed_lr uses .001 throughout; step_lr uses .001 epochs1–30/.0003 epochs31–60. Absolute assignment occurs once before the epoch's first minibatch. No scheduler.step, validation trigger, repeated multiplication, moment/counter reset or rescale. All60 rates and their digest are checked against the fixed tables.

Each completed-epoch last.pt atomically saves model, optimizer, counters, RNG/order, histories, selected references, full schedule, config/statistics/roster/source/split bindings. Epoch0 optimizer LR .001, used/phase/range null. Epoch30 optimizer LR .001, candidate next LR .0003. Restore all states before assigning .0003 for31; an interrupted31 replays from committed30 with the same transition. Later checkpoints store their actual used rate and next rate. Production validators require exactly completed_epoch×1881 steps,135 state entries and exact scalar counters; completed60 cannot resume. History explicitly records global update start/end.

last.pt is the sole current commit; best.pt is a selected snapshot, not the current resume point. A stopped incomplete epoch is replayed from the preceding complete commit. Exact RNG/order checks and fixed numerical replay tolerance do not promise bitwise complete CUDA trajectories. Do not use technical states or old trained weights; production rebuilds known fresh base/full hashes and verifies its source-bound roster before update.

Each runs/<arm>/attempts/<id> retains admission XML, stable PID/start/boot/UID/cwd/argv identity, exact command/environment, barrier/registration and logs. Application interruption is not worker failure. Inspect identity/status/terminal first; preserve genuine failures and request explicit reviewed recovery rather than blind restart. Recovery retains source/settings/authority, acquires fresh locks/admission and creates a unique attempt, never reruns FIT_COMPLETE. All checkpoints/predictions remain server-only.

After60/112860, verify60 orders, source/roster/split/access and checkpoint/export provenance. Selection is earliest strict minimum native validation SSE over0–60, all66860 labels. Report selected and fixed60, plus descriptive prefix0–30/postdrop31–60 history optima. A selected prefix checkpoint has not experienced the intervention and cannot alone support schedule benefit or confirmation seeds. The candidate numerical screen remains+.003 above both selected fixed_lr and retained .44716940136585204, with no automatic allocation. No TEST or extra inference. Geometry export uses one checkpoint and supplied(z,pos,batch,n,edge_index), no old training readers/checkpoints.

## Bounded preparation execution

CPU command uses the pinned Python with CUDA empty and OMP/MKL=2 to run synthetic_checks.py once, after CPU_SOURCE_REVIEW PASS. Seven toy calls, zero full-model updates; PARAMETER_ROSTER is immutable and created before optimizer calls. GPU command is the same CPU-prebound Python running run_gpu_preflight.py without a device argument after TECHNICAL_SOURCE_REVIEW PASS. It selects once under locks in order1,2,4,6 and runs exactly six disposable TRAIN updates with fixture labels30/31/31 and actual counters1/2/2. Results are GPU_PREFLIGHT.json and ops/gpu_preflight_attempt/COMPLETE.json; absence of a result is not retry permission. A HOLD or failure is preserved for explicit review. No raw validation/TEST is opened during preparation.

Completed once: CPU result5a2506a8 (seven toy calls/zero full-model updates), GPU result2bf6eab7 (six discarded TRAIN updates), independent reviews55a20955/09250574. GPU1 selection/registration/exit0 is recorded under ops/gpu_preflight_attempt. No resource failure or model-stage repeat occurred. The reviewer-only selector mock correction remains preserved. Final metadata seal/review is pending, production blocked.
