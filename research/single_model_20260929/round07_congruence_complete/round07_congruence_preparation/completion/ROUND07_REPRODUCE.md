# Round07 reproduction and completed-run records

All three 60-epoch fits and the saved-output analysis are COMPLETE. These commands document the executed recipe; completed-stage guards intentionally reject repeats in this namespace. Do not delete markers, re-evaluate models or score TEST. No completed run needs recovery.

## Exact sources and environment

- Preparation commit: `271d61f6dc9e21009c31d10467fe704e44b974ad`, main and research, parent `68520927333b27c48381401c8516a6709b79c40e`.
- Server directory: `/home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation`.
- Frozen185-file manifest: `68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203`.
- Independent preparation review: `e3ae0804141dc37fdbcb3797411a301e75dabe223eea4005d0c043197f423826`.
- Root execution authorization: `dfb6a5fe356e30da37260830f8032fe7892f2749bb0afc30b6daf277652afaaa`.
- Verified preparation publication receipt: `ce3f342916bb6ecb6a76516bc8da8b64fc1c9839f406dcd9e852c6c12185d6c0`.
- Split: `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent verification: `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`.
- TRAIN statistics: `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.

Pinned Python3.10.19: `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. ENVIRONMENT.json records torch2.5.1+cu121, e3nn0.4.4, NumPy2.2.6, scipy1.15.3, sympy1.13.1, torch-cluster1.6.3+pt25cu121, torch-scatter2.1.2+pt25cu121 and torch-geometric2.7.0. Restore exact dependencies using the preparation archive/restoration mapping and SOURCE_LINEAGE.json. Do not silently substitute later repository code. Private data, split arrays and weights are separate server prerequisites, absent from Git.

## Executed fit command

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /usr/bin/python3 launch.py \
 --authorization /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json \
 --arms original scalar tensor
```

Admission was sequential; the three fits then ran concurrently under separate inherited exclusive GPU locks and worker locks. Each child received its physical UUID before the pinned interpreter imported CUDA modules, so logical cuda:0 maps to its assigned GPU. The stdlib gate binds exact sources, root authority, review, archive-first publication and split before scientific imports.

| Arm | Physical GPU | CUDA_VISIBLE_DEVICES | Original PID / start ticks |
|---|---:|---|---|
| original |1|GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800|1591761 / 1274697650|
| scalar |2|GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa|1591766 / 1274697673|
| tensor |4|GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184|1591771 / 1274697696|

Owned command/environment/boot/UID/cwd/argv hashes, admission XML, logs and registration records are in each run's sole attempts directory. Registrations returned0, and independent startup confirmed actual placement and lock descriptors. No silent fallback, unhealthy GPU3/7 use or unrelated process intervention occurred. All three completed normally without a failure or resume. Process absence is exit evidence, not an observed child OS exit code.

## Fixed scientific recipe

All arms use original PSD MTO16/query32/router128/head128, fresh seed11 and PCG64 order11, TRAIN-only normalization, original LE+Ls with coefficients1, fixed Adam AMSGrad LR .001/betas(.9,.999)/eps1e-8/batch64/WD0/clip5, FP32 without AMP/TF32, two threads and60 epochs. Right-F is disabled/frozen; no decorrelation penalty or raw-f objective is added. Common base hash is231dfaf3… and common full-schema hash e601a037…; no trained or disposable fixture state initializes production.

The shared gate has177 parameters and nine invariant inputs, as frozen in PROTOCOL.md and `congruence.CONTRACT`. With all ten incoming state tensors, B=(sum(A)+1e-6 I/3)/(sum(trA)+1e-6), q=||B||F/sqrt(3), b=.25 tanh(g). Original bypasses/freezes the gate; scalar uses A'=(1+bq)^2 A; tensor uses A'=(I+bB)A(I+bB)^T. Both modified arms train the same gate schema. Predicted E is unchanged forward and remains differentiable in the gate; no labels or masks enter the transform.

Native evaluation f=2 E_eV tr(A')/(3×27.211386245988), calculated in FP64 against authoritative printed raw-f labels. No clamp, teacher energy, calibration, averaging, label matching or exclusions. The selected-row decoder accesses only authorized TRAIN/validation numeric rows; TEST stays sealed. Every validation comparison uses66,860 valid labels. Earliest minimum pooled validation SSE selects from0–60. The tensor gate requires+.003 over both selected contemporary controls and the retained.44716940136585204 reference; it failed.

## Private checkpoints and standalone inference

Each FIT_COMPLETE.json binds112,860 updates, history and all checkpoint hashes. `last.pt` is the committed resume entry with Adam/RNG/order/history, `best.pt` the selected training snapshot and `geometry_best.pt` the standalone export. No recovery is pending and no completed arm may relaunch.

| Arm | Epoch | Standalone relative path | SHA256 |
|---|---:|---|---|
|original|52|runs/original/geometry_best.pt|4140329c1bbd121bfe2db8bac4b053f09aebd14c9808f13f56543f89a7d95b51|
|scalar|49|runs/scalar/geometry_best.pt|46e7754c66ae995d3ab860366b2b08fa98e0deb025405954f3812a55ae9bddae|
|tensor|34|runs/tensor/geometry_best.pt|18d3b15914fa34b217ac46a0a5a5b5691ddde808450f9a706791fa80d0b363b4|

The stronger retained v2 reference is `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`, selected epoch45 R².44716940136585204. Use its matching Round05 source/format, not the Round07 loader.

For a Round07 artifact, `predictor.load_predictor(path, device='cpu')` reconstructs one model from embedded config/TRAIN statistics/full tensors/readout mode and verifies the exact transform contract plus state/buffer fingerprints. Call `model(z,pos,batch,n,edge_index)` with molecule count n and optional geometry edges. It returns predicted E/A'; derive f with the fixed conversion above. No QC label, old checkpoint, raw dataset, cache or external normalization is required. This documents deployment; no additional terminal model forward was executed.

## Executed terminal analysis

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python terminal_analysis.py \
 > ops/terminal_analysis_01.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /usr/bin/python3 write_completion_report.py > ops/write_completion_report_01.log 2>&1
```

The analysis ran once and refuses a repeat after ANALYSIS_RECEIPT.json. Six saved validation sets (selected/fixed60 for three arms), CPU checkpoint payloads, validation identity metadata and the published retained aggregate were used. All source/checkpoint/prediction hashes, optimizer/RNG/order/selection contracts and pooled/state/tail/bin arithmetic passed. Export tensors exactly equal selected checkpoint tensors and their mode/contract match. Stored buffer fingerprints are present; they were not reconstructed through new model loading. Preparation already tested strict one-checkpoint geometry parity.

No model instance/forward, raw-dataset target or TEST decoding occurred. The imported congruence module supplies frozen contract definitions only. Gate diagnostics come from saved pre-update TRAIN trajectory logs; within-epoch parameter displacement is not cumulative movement. False-bright bin membership depends on each model's predictions. Bootstrap uses2,000 component draws/seed20260930 with recomputed pooled SST, all states together; intervals are conditional descriptive validation evidence, not seed confirmation. No paired interval versus the aggregate-only retained reference.

For an independent reproduction, use exact private inputs/hashes in a separately reviewed output workspace. Do not remove receipts or overwrite completed evidence here. All tensors, data, predictions, identities and caches remain server-side. TERMINAL_MANIFEST.json is the explicit lightweight payload list; private paths/hashes in ANALYSIS_RECEIPT are provenance only.
