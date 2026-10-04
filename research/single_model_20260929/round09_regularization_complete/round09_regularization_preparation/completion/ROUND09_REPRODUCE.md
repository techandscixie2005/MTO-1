# Round09 completed-run reproduction

These commands document the fixed experiment. Completed fits and analysis must not be repeated in this namespace. No TEST scoring or recovery is pending. All private prerequisites and trained artifacts remain on the server.

## Pinned recipe and provenance

- Server namespace: `/home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation`.
- Preparation main/research commit: `566aa5e02c2f7cad6dc8c7b8c732ad45dfb9c8be`.
- Frozen360-file manifest: `52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930`.
- Independent preparation review: `08d0c6e804d50b019f027b05282fbb1168d3c7ec59aa0e59b6068ddf0e5225cb`.
- Preparation publication receipt: `440d25926c543a7a03089f8515135a124f56851ecf84e3464b20cd07561efd79`.
- Exact production authority: `fc16c206e829c345098fcd853febcd2f83c88a56c3ea92f5506c0edd67678765`.
- Split: `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent verification: `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`.
- TRAIN statistics: `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.
- Ordered parameter roster: `290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943`.

Restore exact code/dependencies using FROZEN_MANIFEST, SOURCE_LINEAGE and the preparation archive restoration mapping. Pinned Python is `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`; ENVIRONMENT.json records Python3.10.19, torch2.5.1+cu121, e3nn0.4.4, NumPy2.2.6 and the geometry/scatter dependencies. Installed Adam/clip source hashes are bound. Do not substitute later optimizer semantics silently.

## Executed launch

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python launch.py \
 --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms zero_decay coupled_l2
```

Fresh admissions are sequential; the two fits then run concurrently. Children receive their exact physical UUID before imports, with logical cuda:0 mapping to that GPU. `registered_entry.py` holds the same owned PID behind a pipe barrier until registration succeeds; only then does it execute `train.py`. Both shared GPU and worker locks remain inherited. No automatic fallback or retry.

| Arm | GPU / UUID | PID / start ticks | Attempt |
|---|---|---|---|
| zero_decay |1 / GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800|2004444 / 1292019213|round09_zero_decay_1791072527376721902|
| coupled_l2 |2 / GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa|2004449 / 1292019237|round09_coupled_l2_1791072527606971050|

Both original workers exited after normal FIT_COMPLETE markers. Process absence is exit evidence, not an observed OS child exit code. Unique attempt directories retain launch/admission/registration/train logs; ops/PRODUCTION_LAUNCH_01_RECEIPT records the exact invocation. Independent startup observed actual UUID placement, environment, locks and barriers. No unrelated process was signaled.

## Scientific settings

Both arms use unchanged original PSD MTO, original LE+Ls, seed/order11,60epochs/batch64/1881batches per epoch,112860updates, fixedLR.001/AdamAMSGrad/betas(.9,.999)/eps1e-8/clip5. No scheduler, early stopping, AMP, TF32, teacher energy, calibration, active right-F/decor/congruence/transport or trained initialization. Common base hash `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; full schema hash `f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3`.

One ordered135-tensor group has1,552,092 values, including every original trainable bias/norm/embedding/offset/radial parameter. Exactly11 frozen right-F/transport tensors and all buffers are excluded. No bias or normalization exemptions. Only weight_decay changes:0 versus1e-4. foreach/fused=None, capturable/differentiable/maximize=false. After task backward and clipping, Adam adds lambda*p internally before its moments; this is coupled L2 rather than AdamW. grad=None skips parameter/state/decay, whereas a zero gradient does not. No second clipping step.

TRAIN statistics and target thresholds are fit only on newTRAIN. Selected-row readers avoid numeric TEST target decoding. All66,860 validation labels and zeros remain included. Native f is FP64 `2*E_eV*trace(A)/(3*27.211386245988)`. Earliest strict-minimum pooled native validation SSE across0–60 selects; fixed60 remains a separate comparison. Promotion requires coupled_l2 to exceed both current selected zero_decay and retained.44716940136585204 by at least.003; it fails.

## Standalone checkpoints and recovery

| Arm | Selected epoch | Relative standalone path | SHA256 |
|---|---:|---|---|
| zero_decay |39|runs/zero_decay/geometry_best.pt|7443fd31430f1539cd6106a675c2dec2f4b1b3783c20e6f1a3dd569a3ae81435|
| coupled_l2 |56|runs/coupled_l2/geometry_best.pt|29c8834ac59fe5986fc0b80dfb50b769358535e33d71bc5293037cb3585d860e|

`predictor.load_predictor(path, device='cpu')` loads the matching Round09 format, embedded original mode/transport contract, config, TRAIN statistics, tensors and strict buffer/tensor fingerprints. `model(z,pos,batch,n,edge_index)` then takes geometry and molecule count; it needs no old checkpoint, dataset, QC labels or cached predictions. This is deployment documentation, not a terminal inference run.

`last.pt` is the complete-epoch resumable transaction with model/optimizer/RNG/order/history/selection references and all scientific pins. `best.pt` is a selected snapshot, not the runner's current commit point. Both fits are complete; never remove FIT_COMPLETE to relaunch. Preserve an actual failure and obtain a concrete reviewed recovery decision before resuming any interrupted run. Preflight-trained states cannot enter production.

The retained reference remains Round05 control45 at `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`. Use its matching Round05 loader; no averaging.

## Saved-output analysis

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round09_regularization_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python terminal_analysis.py \
 > ops/terminal_analysis_01.log 2>&1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /usr/bin/python3 write_completion_report.py > ops/write_completion_report_01.log 2>&1
```

The source-reviewed CPU analysis runs once and refuses completed ANALYSIS_RECEIPT. It checks source/authority, optimizer roster/options/moments,60orders/history/selection, source/exits/access and export equality. It reads four saved selected/fixed60 validation sets, CPU checkpoint contents, validation component metadata and the previously published retained aggregate. No model construction/forward, raw dataset target or TEST array is used. All four truth/mask sets agree. Stored buffer fingerprint format is checked; strict buffer reconstruction/parity was established in preparation.

The component bootstrap uses2000 paired draws/seed20260930, sampling whole validation components with all molecules/states and recomputed pooledSST. Draw molecule counts can vary. Intervals condition on reused validation and selected checkpoints, not seed/model-selection uncertainty. No paired interval versus the aggregate-only retained reference. Raw-f/state/tail/bin/energy/error-concentration and regularization plots all use saved outputs or recorded aggregates.

The report renderer opens aggregate JSON only. ANALYSIS_RECEIPT binds exact source/input/output hashes. TERMINAL_MANIFEST.files is the explicit lightweight payload allowlist; private input hashes are provenance only and must not be expanded into checkpoint/data/prediction downloads. Exact independent reproduction needs private prerequisites in a separately authorized workspace, not removal of completed-stage guards.

The five toy calls and six discarded TRAIN technical updates were engineering checks. Original GPU1 preflight admission failed before any child/data/update; its empty attempt and snapshot remain preserved. The reviewed GPU4 retry completed once. No scientific settings or tolerances changed, and neither technical state was reused. These preparation records remain inherited from the verified preparation archive.

The analysis completed once, exit0 in4.00seconds, under source review41e7ae2ba01e7cf005e69da8a8b35da8c9089945f766c07017d7dc096b4c966c. Results SHA394bc58c2e3996432b6b7f94ed74c8afa285b65ac4b85fb8d72a086b691f830f; ANALYSIS_RECEIPT SHA11fbcc152e020c44125563bbab6b567b698ed3e93a6a3e464a3cc5c284b4a2ce. The invocation receipt preserves exact command/environment/exit/log. No numerical rerun is needed.
