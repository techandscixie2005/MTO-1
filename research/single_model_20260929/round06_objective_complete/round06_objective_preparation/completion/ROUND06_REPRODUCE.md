# Round06 reproduction and recovery records

Both60-epoch fits and the saved-output analysis are COMPLETE. Commands below document the executed recipe; completed-stage guards intentionally reject repeats in this namespace. Do not delete terminal markers, re-evaluate models or score TEST. A later fit requires a new reviewed root decision.

## Exact source and environment

- Preparation commit: `c26a860b972a1c01267f8b766393772173515b92`, both main and research, parentf75a23c.
- Server directory: `/home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation`.
- Frozen134-file manifest: `4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b`.
- Independent preparation review: `8d2dd32e954f533bdcdf298693b7d2c457eef5fb123824c3eeaddf026e33a986`.
- Root execution authorization: `2c6c7f75993719fb5386df162248461cd7355bd2f033ae0bfd2848c146d9d41c`.
- Preparation publication receipt: `43d63a5d6bd1bbd7d4f8b20fc8b0579511f9920a16987a4b2d2366a2000ec07a`.
- v2 split: `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent verification: `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`.
- TRAIN statistics: `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.

Pinned Python3.10.19: `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. ENVIRONMENT.json records torch2.5.1+cu121, e3nn0.4.4, NumPy2.2.6, scipy1.15.3, sympy1.13.1, torch-cluster1.6.3+pt25cu121, torch-scatter2.1.2+pt25cu121 and torch-geometric2.7.0. Restore exact dependency paths using preparation archive snapshots/restoration mapping and SOURCE_LINEAGE.json; do not substitute later repository code silently. Dataset/private split/weights remain separate server prerequisites, absent from Git.

## Executed fit command

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /usr/bin/python3 launch.py \
 --authorization /home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json
```

The stdlib gate verifies source, root authority, independent review, archive-first remote publication and split before scientific imports. Launcher admission was sequential; the two fits then ran concurrently on separate inherited exclusive shared locks. UUID binding precedes child Python imports, so logical cuda:0 maps to the intended physical device:

| Arm | Physical GPU | CUDA_VISIBLE_DEVICES |
|---|---:|---|
| trace_control |1|GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800|
| raw_f |2|GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa|

Owned identities/boot/start ticks/environment, resource XML, commands and logs remain in each run's sole attempts directory. Both registrations returned0. Independent startup verified actual placement and locks. Unrelated GPU0 was preserved; no fallback or unhealthyGPU3/7 use occurred. Both completed normally, with no failure or resume. Process absence is terminal evidence, not an observed child OS exit code.

The scientific recipe is common original PSD MTO16/query32/router128/head128, fresh seed11/base hash231dfaf3…, PCG64 order11, new TRAIN-only normalization, Adam AMSGrad lr.001 fixed/betas(.9,.999)/eps1e-8/batch64/WD0/clip5, FP32/noAMP/noTF32, two threads and60 epochs. Both retain dormant frozen F schema with adapter disabled and no decorrelation loss. Only objective changes:

- control: unchanged LE+Ls;
- candidate: LE+mean[(2 E_eV tr(A)/(3×27.211386245988)−printed f)²]/.0025089892829484074.

Coefficient1 remains fixed; gradients pass through predicted E and A. Valid entries precede nonlinear arithmetic, valid zeros stay included, no target-E substitution/detach/calibration/clamp/label exclusions. Training f is FP32; reported raw-f uses FP64 arithmetic and printed targets. All source/order hashes and the original control-loss parity are preserved in preparation receipts. Earliest minimum all-label validation SSE selects among epochs0–60. No TEST access or automatic extension/seed allocation.

## Completed private checkpoints

Each FIT_COMPLETE.json binds112,860 steps, history, last.pt, best.pt and geometry_best.pt. `last.pt` is the committed full recovery entry with Adam/RNG/order/history; `best.pt` is the selected training snapshot. No completed run requires recovery. Technical fixture states never initialized production.

| Arm | Epoch | Standalone checkpoint | SHA256 |
|---|---:|---|---|
|trace_control|49|runs/trace_control/geometry_best.pt|7d3105fd3e32789baafb3c7cb8f63880a8928dd28d1d3bf04db2c1cc2cef2a61|
|raw_f|18|runs/raw_f/geometry_best.pt|cb8d8e82edff74a45b64293d1b03537df0f2b75037b96e45e417e1e904ca7e29|

Neither replaces the stronger retained v2 reference: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`, selected epoch45 R².44716940136585204. Use the matching round's predictor/source closure for each format tag.

`predictor.load_predictor(checkpoint_path, device='cpu')` reconstructs one full geometry predictor from embedded config/stats/model and checks exact state/buffer fingerprints. It accepts geometry inputs (atomic numbers, coordinates, molecular batching and geometry edges), then returns E/A; compute f with the same conversion above. No original checkpoint, QC labels, dataset or external statistics are required at inference. These are documented inference commands, not an additional terminal replay; this closeout performed no model forward.

## Executed terminal analysis

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python terminal_analysis.py \
 > ops/terminal_analysis_01.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /usr/bin/python3 write_completion_report.py
```

The analysis completed once and refuses repeat execution after ANALYSIS_RECEIPT.json. It used four saved validation prediction sets (selected/fixed60 for both arms), private CPU checkpoint payloads, validation component/index metadata and the previously published Round05 aggregate. It verified every current source pin, checkpoint/array hash, optimizer/RNG/order/selection contract and all-state/tail/bin arithmetic. No new model, raw dataset target or TEST inference/decoding occurred. Bootstrap seed20260930,2,000 component draws; conditional descriptive intervals only. No paired interval against the aggregate-only old reference. No predictions are averaged.

For independent reproduction, use the exact private inputs/hashes in a separately reviewed output workspace. Do not remove this namespace's receipts or overwrite evidence to rerun. All weights, optimizer states, datasets, predictions, identities and caches stay server-only. The lightweight terminal manifest lists only approved text/SVG/code/log records; private paths in ANALYSIS_RECEIPT are provenance references, not download requests.
