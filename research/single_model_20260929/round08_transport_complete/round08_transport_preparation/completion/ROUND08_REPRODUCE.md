# Round08 reproduction and completed-run records

All three fixed60 fits, first-epoch inspection and saved-output analysis are COMPLETE. Commands below document the executed recipe. Guards intentionally reject repeats in this namespace; do not remove markers, rescore models or open TEST. No recovery is pending.

## Exact sources and prerequisites

- Preparation commit: `a10eb9a35b26014f00138c61476b33e9070bb69b` on main and research, parent `703c2cd7c68f4068160ab9d3dbbab64d4d5af6ba`.
- Server namespace: `/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation`.
- Frozen258-file manifest: `de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146`.
- Independent preparation review: `87de2d873f2a24033a2f646ac825f26abbbae7eadbd5305a723fbb6a17093e8b`.
- Root execution authority: `8da6f43114575a73ff3e7dba918ec6ade913ada3c53d3aafb00b4e4e1534f4a6`.
- Verified preparation publication receipt: `44c7a8fcc94a004ab1d973be76c4c79a3ed25d46307c73e12211935e7902c8e9`.
- Split: `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent verification `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`.
- TRAIN statistics: `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.

Pinned Python3.10.19 is `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. ENVIRONMENT.json records torch2.5.1+cu121, e3nn0.4.4, NumPy2.2.6, scipy1.15.3, sympy1.13.1, torch-cluster1.6.3+pt25cu121, torch-scatter2.1.2+pt25cu121 and torch-geometric2.7.0. Restore exact dependency sources through the preparation manifest/restoration mapping and SOURCE_LINEAGE.json. Do not silently substitute later code. Private raw data, split arrays and weights remain separate server prerequisites and are not in Git.

## Executed launch recipe

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python launch.py \
 --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms original local neighbor
```

The reviewed launcher admits sequentially, then three fits run concurrently under separate shared GPU locks and worker locks. UUID visibility is prebound before child imports, so logical cuda:0 maps to the assigned physical GPU. Each child initially runs `registered_entry.py train.py --arm ARM --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json`; the same owned PID waits for the parent's exact registration-success payload before importing/running the scientific script. Registration failure closes the barrier and stops only the owned child. There is no automatic retry/fallback.

| Arm | Physical GPU | CUDA_VISIBLE_DEVICES | Original PID / start ticks |
|---|---:|---|---|
| original |1|GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800|1673191 / 1283342887|
| local |2|GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa|1673196 / 1283342910|
| neighbor |4|GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184|1673201 / 1283342933|

One attempt per arm preserves LAUNCH_RECEIPT, GPU_ADMISSION.xml, REGISTRATION_TOOL.log and train.log. Boot, UID, cwd, argv, environment, exact authority and registration barrier are bound. Startup independently observed UUID placement and GPU/worker locks. All runs completed normally without resume; process absence is exit evidence, not an observed child OS exit code. GPU0/unrelated jobs were not owned; GPU3/7 were excluded.

## Frozen scientific recipe

Original PSD MTO16/query32/router128/head128 with core128/three blocks, fresh seed11 and PCG64 order11, TRAIN-only normalization, original LE+Ls, Adam AMSGrad fixed LR.001/betas(.9,.999)/eps1e-8/batch64/WD0/clip5, FP32/noAMP/noTF32/two threads,60 epochs. No scheduler, raw-f objective, decorrelation penalty or trained initialization. The common fresh base hash is `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; common full schema is `f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3`. The disposable nine-update fixture never initializes production.

All arms retain the same 768-theta checkpoint schema in blocks2/3. Original freezes/bypasses theta; local/neighbor train it from zero. For operational receiver j=index[1] and source i=index[0], G=tanh(existing mijs2), H_j is the incoming-degree mean of G*T_j (local) or G*T_i (neighbor), and delta_j=tanh(theta)*H_j. T is pre-block and each coefficient is shared across magnetic components within an actual irrep/channel slice. Add after original outt(scatter(mijt)) and before local uattn; Attention is evaluated once. Block1 and the original PSD head remain unchanged. Exact formula/layout/empty-neighborhood behavior is embedded as `transport.CONTRACT` and checked on load. Right-F remains disabled/frozen.

Native raw-f evaluation is `2*E_eV*trace(A)/(3*27.211386245988)` in FP64 against authoritative printed raw-f labels. Softplus E and PSD A preserve nonnegativity. No teacher energy, calibration, clamp, averaging, permutation or exclusion. Selected-row readers decode only authorized TRAIN/validation numeric rows; TEST remains sealed. Each validation uses66,860 labels. Earliest minimum pooled native validation SSE across0–60 selects one checkpoint. Neighbor needed+.003 over both selected current controls AND retained.44716940136585204; the screen failed.

## Private checkpoints and standalone inference

Each FIT_COMPLETE binds60 epochs/112,860 updates/history and checkpoint hashes. `last.pt` is the committed resumable entry with optimizer/RNG/order/history; `best.pt` is the selected model/optimizer snapshot and `geometry_best.pt` the standalone predictor. No completed run needs recovery or may be relaunched here.

| Arm | Epoch | Standalone path relative to namespace | SHA256 |
|---|---:|---|---|
| original |36|runs/original/geometry_best.pt|7fe3f2562f39afca83d353995a9412c069c82414923422a0f95f9304fd6382f1|
| local |30|runs/local/geometry_best.pt|58895cae3aed9d536c93d307ae02e0d862ef1b07c0e2689890a11b9f944770a8|
| neighbor |23|runs/neighbor/geometry_best.pt|470fa5676c652d8c4e11ea24c68d2b5c8d00720bdcd068a6b075a580b95f53a6|

For a Round08 artifact, call `predictor.load_predictor(path, device='cpu')`, then `model(z,pos,batch,n,edge_index)` with molecular geometries, molecule count n and optional geometry edges. The loader uses embedded model config, TRAIN statistics, full weights, transport mode and contract, with strict tensor/buffer verification. No old checkpoint, QC labels, dataset or cache is opened. Predicted E/A yield f by the conversion above. This documents deployment; no terminal forward/reconstruction was performed.

The stronger retained checkpoint is `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`, epoch45 R².44716940136585204. Use its matching Round05 format/source, not the Round08 loader.

## Executed saved-output analysis

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python terminal_analysis.py \
 > ops/terminal_analysis_01.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /usr/bin/python3 write_completion_report.py > ops/write_completion_report_01.log 2>&1
```

The analysis ran once after source review46108a5b…, refuses a repeat after ANALYSIS_RECEIPT, and binds source/input/output hashes. It reads selected/fixed60 saved validation arrays for three arms, CPU checkpoint contents, validation component metadata and the published retained aggregate. It checks source/authority/order/optimizer membership/steps/selection/access, identical truth/masks, complete pooled/state/tail/bin arithmetic, mode/contract and selected export tensor equality. Stored buffer fingerprints exist but were not reconstructed; strict standalone parity was established in preparation.

No model instance/forward, raw dataset targets or TEST array was opened. Importing transport supplies contract definitions only. Logged transport activity is pre-update TRAIN trajectory evidence; regularized norm denominators use max(norm,1e-12), with zero/below-floor counts. The same logs supply plots; no additional inference. Checkpoint theta statistics are aggregates only, never coefficient arrays. Bootstrap uses2,000 paired identity-component draws, seed20260930 and resampled pooled SST with all states together. Its intervals are conditional on reused validation/checkpoint selection, not seed confirmation. No paired interval versus the aggregate-only retained reference.

Analysis outputs: ROUND08_RESULTS.json SHA `5ba356f7e37def370d09492855378b580cf5fa5e66afb4db990d7b3a5cc0dd58`; ANALYSIS_RECEIPT SHA `8451c3a05efb9b149a8e5403f389607dc125094ef79f37ca57617e5736ab727d`. Figures contain aggregate values only. The stdlib report renderer reads that JSON alone.

An independent reproduction requires the exact private prerequisites in a separately reviewed workspace; never remove immutable receipts or overwrite evidence. All weights, optimizer tensors, coefficients, datasets, identities and prediction arrays remain server-side. TERMINAL_MANIFEST.files is the exact lightweight archive payload list. Opaque input hashes are provenance references and must not be expanded into downloaded payloads.
