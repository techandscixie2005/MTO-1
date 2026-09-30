# Round05 production — COMPLETE; closeout publication pending

Terminal update at the 15:56 UTC heartbeat: every arm has FIT_COMPLETE.json with 60 epochs and 112860 updates, no FAILED.json, and the original processes are absent. QC independently confirmed registry/resource terminal state in campaign monitoring/SCHEDULED_ROUND05_TERMINAL_20260930T1555.json (SHA 91fce5b8263ad754c34dee18efba12092d645cc230ec40b6f64c84a42f791b20). Do not launch or resume any arm. CPU-only verification and saved selected/fixed60 validation analysis completed successfully; independent scientific review passed. No model inference, TEST access, extension or new seed is authorized.

Verified selected R²: control .4471694014 at epoch45; adapter .4150203688 at41; decorrelation .4123991927 at23; both .4249660535 at20. The +.003 gate fails. Root ROUND05_DECISION.md closes this factorial, retains control45 as the v2 reference, and authorizes D-first closeout publication by QC. completion/ROUND05_RESULTS.json SHA bf6920dd8c883460ecb83db3e44ec3debe98152a3f1521c869a5102b4203eb24 and ANALYSIS_RECEIPT.json SHA 31bc94680894143e8171033e80f6fe3de5e65bc986ffd74462bc66ea08a7fd88 bind saved metrics and all source/checkpoint/order/access verification. Complete report, reproducible commands and private checkpoint paths are in completion/. All arrays/weights stay server-only. The earlier startup chronology below is historical.

Current ownership: QC archives/publishes the reviewed closeout; history binds final narrative/terminal manifest; science is available for bounded corrections. After archival, root selected proposal development only for an unchanged original-PSD LE+Ls versus normalized raw-f objective comparison. No next code, technical optimizer updates or production fit is authorized. Deferred tensor-congruence QC remains a separate hypothesis. The four-hour heartbeat must never restart or rescore these completed runs.

Science is the sole executor. Read actual registered identities and terminal markers before any action; interruption of this conversation is not experiment failure. Do not rerun completed preparation or Round04.

## Authority and launch

- Preparation published to main and the research branch at `9de80e37e1586c2d3068b188ed2c242ee9883147`.
- Publication receipt: campaign `ops/ROUND05_PREPARATION_PUBLICATION_RECEIPT.json`, SHA `fbdcbb4ce7e2b86ae6c906affeb4e7c08a57fc9e48e58a60d485dd61c7a4eba5`.
- Exact root authorization: `PRODUCTION_EXECUTION_AUTHORIZATION.json`, SHA `07308b113557d2196cefc951a603f3f1a5093009eaae1576412f1e393bb8162c`.
- Frozen manifest: `66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1`; independent review: `ef7d90250fd0491fba1432201416de983afa86ee6ae7ebf6e086735f5e10ff19`.
- The stdlib execution gate verified every binding and all 82 source hashes before launch. No run directories existed. Reviewed launcher exited 0, all four fresh GPU admissions and registrations returned 0.

Command from `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation`:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 /usr/bin/python3 launch.py --authorization /home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json
```

The launcher prebound each child's GPU UUID before imports and held separate GPU locks. Admission was sequential; workers run concurrently.

| Arm | Physical GPU | PID | Start ticks | Registered attempt |
|---|---:|---:|---:|---|
| control | 1 | 1328593 | 1261898113 | round05_control_1790771316371909087 |
| adapter | 2 | 1328598 | 1261898136 | round05_adapter_1790771316605474432 |
| decorrelation | 4 | 1328603 | 1261898159 | round05_decorrelation_1790771316835845292 |
| both | 6 | 1328608 | 1261898182 | round05_both_1790771317058319159 |

Boot ID: `793423dc-86b6-4fb9-9a9d-e74cfa8135a3`. Each exact identity, argv hash, UUID, environment, admission XML and log is under `runs/<arm>/attempts/<attempt>/`. QC owns one independent startup check and the unchanged four-hour monitoring schedule; science owns first-epoch resumable metadata inspection.

All four first epochs are now committed and read-only metadata verification passed in `ops/FIRST_EPOCH_METADATA.json`. Durations including validation: control 162.67 s, adapter 166.80 s, decorrelation 162.90 s, both 168.07 s. Every arm has 1881 committed updates, valid full Adam/RNG/order state, the frozen initialization and order hashes, and correct selected-version hashes. The common epoch-0 R² differs by less than 2e-10 across CUDA executions; no bitwise trajectory claim is made. All production data ledgers match only TRAIN and validation. No failure marker was present. The first inspector invocation was slightly early and stopped on the epoch-1 readiness assertion; `FIRST_EPOCH_INSPECTION_NOTE.md` preserves that fact. No production action was repeated or altered.

Launch occurred at server UTC 2026-09-30 12:28:36. At first-epoch pace, expected completion is approximately 15:12–15:17 UTC (23:12–23:17 Asia/Shanghai), subject to system load. No decisions are made from early scores. The fixed 60-epoch recipe continues.

Independent startup confirmation passed: campaign `monitoring/ROUND05_STARTUP_CONFIRMATION_20260930T1229.json`, SHA `0ba8a3784e8386b78218372a56613b8b5b5316c90d6dc63bb363755b3e7e11cb`. It verified complete process identities, actual GPU UUID placement, environment and inherited exclusive locks. Its first child-PID-only lock assertion was corrected to inspect inherited fdinfo because /proc/locks retains the launcher PID; no intervention occurred. The existing four-hour heartbeat and monitoring handoff are updated. Root accepted first-epoch evidence; no active polling or extra evaluation is required. The next heartbeat should inspect actual terminal/process state and coordinate independent terminal analysis and D-first archiving.

## Fixed work and recovery boundary

All four arms independently start from the frozen fresh seed 11 tensors and use the new TRAIN statistics, common 60 epoch orders, original LE+Ls, fixed LR .001, batch 64, Adam AMSGrad, WD 0, clip 5, FP32. The decorrelation strength is .001 in its two arms. Epoch 0 remains eligible; earliest minimum pooled validation raw-f SSE selects one checkpoint per arm. No ensemble, test scoring, extension, additional seed or new model is authorized.

The production reader may now decode the frozen new TRAIN and validation partitions only. New TEST targets remain sealed. The new split uses historically exposed molecules and is not external fresh confirmation.

Monitor `status.json`, `history.jsonl`, `FIT_COMPLETE.json` and `FAILED.json` under each run. All four must reach 60 epochs and 112860 updates before round completion. Preserve full logs and checkpoints. An actual owned failure requires diagnosis and an explicit reviewed resume; never blindly relaunch or signal an unrelated process. `last.pt` is the atomic completed-epoch recovery point with model, optimizer, RNG, order state and immutable selected-version hashes. An interrupted partial epoch replays from the preceding committed epoch. `best.pt` is a selected snapshot; `geometry_best.pt` is exported at normal completion. Technical preflight tensors must never enter production.

After terminal completion, analyze existing selected and fixed-60 records under the frozen rules, independently review, then download lightweight records to D:\MTO\archives before commit/push. All tensors, raw labels, identities, predictions and caches remain server-only. No automatic post-fit extension or TEST evaluation.
