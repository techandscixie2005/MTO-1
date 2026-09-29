# MTO single-model oscillator-strength research

## Verified result through Round03

The strongest eligible recipe by pooled validation raw-f R² remains the original eta0 model plus its historical fixed affine calibration, **.41811924**. It uses one model and one composite checkpoint. Its validation calibration and historical test scores were already exposed; there is no fresh-holdout confirmation.

Round03 produced a second verified composite recipe: an affine map fitted to predictions from a clean internal-TRAIN heldout source, then transferred to the original full eta0. It reaches **.41637451**, improving the native anchor by .01108039 and the matched in-sample affine by .01435016. It passes the prespecified preparation-allocation gate but does not replace the incumbent. Its MAE and bright-tail errors are better than the historical calibrated recipe and worse than native. Independent seed confirmation remains outstanding.

| Eligible recipe | Validation pooled raw-f R² | Historical exposed-test R² | Main qualification |
|---|---:|---:|---|
| Native eta0, seed11 epoch33 | .40529412 | .45581511 | Original single-model anchor |
| Historical fixed affine eta0 | .41811924 refit; .41665766 historical OOF | .45966723 | Highest eligible recorded score; calibration used outer validation |
| Round03 heldout-source affine eta0 | .41637451 | Not evaluated | Both coefficients frozen before this outer evaluation; outer set still reused |
| Historical frozen latent-h residual | .40678990 | Not evaluated | Small early gain, later overfit; separate historical storage was not packaged here |

Commit69bf39f's F5 validation .48769961/test .50862490 averages five predictions and is ineligible. No ensemble is used or proposed as the final predictor.

## Checkpoints and geometry-only inference

**Incumbent:** `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`

SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.

Formula: `max(0, .8511830211044088*f_native + .003725185373211049)`.

**Round03 transferred alternative:** `/home/inspur/MTO-1/research/single_model_20260929/round03_transfer/affine/heldout_source.pt`

SHA256 `291b80c87afc0c841781d3c7e1b376f6008a13d955291110d5305280d75df398`.

Formula: `max(0, .9025853252242239*f_native + .0018285582112617521)`.

Both checkpoints embed the same 1,552,092-parameter eta0 model, configuration, original training normalization and two fixed FP64 buffers. The loaders need pinned implementation/dependencies but do not reopen a second trained checkpoint or QC labels. The temporary source is absent from deployed inference.

```python
import sys
import torch
sys.path.insert(0, '/home/inspur/MTO-1/research/single_model_20260929/baselines')
from calibrated_eta0 import load_predictor

model = load_predictor(
    '/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt',
    device='cpu')
with torch.no_grad():
    outputs = model(z=z, pos=pos, batch=batch, n=n, edge_index=edge_index)
f = outputs['f']
```

For the transferred alternative, import `load_predictor` from `round03_transfer/predictor.py` and use its checkpoint path. Inputs are atomic numbers, Å positions, molecule membership/count and geometry-derived edges. Ten ordered outputs are returned per molecule; no quantum-chemical labels are required. Native f is `(2/3)*(E_eV/27.211386245988)*trace(A)`. Affine f is an independent calibrated output: the returned unchanged E/A need not satisfy this identity. Do not silently substitute a tensor rescaling.

CPU access preflight verified self-contained inference; the incumbent has recorded full validation parity. Round03 standalone exports each replayed the same first64 validation geometries against the fixed formula within prespecified tolerances. The reviewer reused that evidence and checked checkpoint tensors/buffers on CPU without new inference.

## Fixed evaluation and evidence boundaries

- Original splits remain120,355/6,686/6,686 molecules. All valid printed raw-f labels, including zeros, are retained. No score-driven exclusions or new outer split were introduced.
- Pooled R² is computed from flattened valid molecule-state pairs, not averaged state R². Current outer validation has66,860 labels and SST158.4062371581119.
- Current bright-tail thresholds .0549/.2412 come from TRAIN and yield6,650/639 validation labels. Historical cutoff .0546/.2377 reports are kept separate.
- No current round evaluated test predictions. Historical test values in the table are reused evidence. No verified fresh holdout exists, and independent seeds cannot create one.
- Split SHA `141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca`; dataset SHA `be8fadca203429575d70642b617730592693be40858dca7189c098a671596330`; raw-label SHA `621dc8723fb6dc96681851a26e4f52fcb7747a4778e57a249a75d22f3af4c632`.

## Controlled experiments

**Round01:** full-model 20-epoch continuation at1e-5, original LE+Ls, same source/order. Control, shared right-CG F, weak raw-M decorrelation λ.001, and both all selected epoch0. Final native R² was about .3662–.3667. Both interventions moved as intended, but neither improved selection. Some MAE/energy/true-tail metrics improved while false-bright/S7/S8 SSE worsened. This is not evidence of uniform degradation or a demonstrated physical mechanism.

**Round02:** all original model parameters/buffers frozen; only4,016 F parameters trained. Matched original LE+Ls versus LE plus raw-f MSE divided by TRAIN variance .002510981243894732, coefficient1; 20 epochs at1e-5. Best native R² .40540783/.40538752, both far below the .003 promotion threshold. Fixed20 .40312554/.40215798. No seed confirmation followed. Historical frozen latent-h residual training already used normalized raw-f MSE and gained only .001496; freezing or raw-f supervision alone is not a new rationale.

**Round03:** original random MTO trained on96,284 internal-TRAIN molecules for exactly33 epochs,49,665 updates, no heldout/outer checkpoint selection. Whole-group split seed20260930 held24,071 molecules; unresolved identity groups stayed in source fitting, all original TRAIN rows were assigned once. Fit-only E/A statistics and energy initialization were recomputed; no pretrained/full-TRAIN initialization was loaded. Source used original LE+Ls, Adam AMSGrad1e-3, batch64, clip5, weight decay0, FP32, AMP/TF32off, and fixed learning rate. The sum of the 33 recorded epoch times was 65.36 minutes.

Two FP64 pooled affine maps used the identical240,710 held labels, including4,546 zeros: one on full eta0's in-sample predictions, the other on the clean source's heldout predictions. Both maps were frozen before outer validation and applied to the same native full eta0. Source quality differs: corresponding uncalibrated calibration-row R² values .66106654 and .42940812. Training size, optimization history and source errors remain confounders.

| Round03 predictor | Pooled R² | MAE | q90 RMSE | q99 RMSE |
|---|---:|---:|---:|---:|
| Native | .405294124 | .017785608 | .095359275 | .219912917 |
| Historical affine | .418119244 | .018219066 | .098349265 | .231384276 |
| In-sample affine | .402024351 | .017819879 | .095038371 | .218778366 |
| Heldout-source affine | .416374515 | .017805840 | .097485595 | .227281574 |

Heldout-source calibration improves eight state R² values versus native; S1 and S5 worsen. S7 improves .04994014→.10010615; S8 barely changes. Fixed true-q99 SSE rises30.9031→33.0088 while its true-dim complement falls63.3020→59.4411. Energy outputs remain identical, with MAE .0913883eV. `ROUND03_RESULTS.json/md` preserve every state's SSE/RMSE/MAE/R² and all-label brightness partitions.

## Reproducible commands and retained source

Pinned server Python:
`/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`.

Round03 frozen manifest SHA:
`93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37`.

Original eta0 model checkpoint:
`/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`, SHA `9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136`.

Original architecture/settings are pinned in `experiments/qm9s_eta_Ef_20260926/configs/mto_eta0.json`: channels16, query32, router/head128, seed11, original LE+Ls, AMSGrad1e-3, batch64, clip5, weight decay0. The historical full run used its validation scheduler/stopping and selected epoch33. Round03 uses the separate fixed33 protocol rather than repeating that selection.

Actual reviewed source launch:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round03_transfer
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python launch_round.py \
  --review INDEPENDENT_PRELAUNCH_REVIEW.json \
  --publication-receipt ../ops/ROUND03_PUBLICATION_RECEIPT.json
```

The source's resumable `runs/source33/last.pt` SHA is `52f2884b4869ae87d85ba56875021ff3d42383cd53bfad356362d213883499c9`. Fixed `source_final.pt` SHA is `769026edf4e999799dd201c3e8f2ad3393e25ef9b3a5a99b23f7ce59ffc984ec`. Source-final is a calibration data generator, not a deployed ensemble member. Resume restores exact order/RNG; the GPU next-update preflight agreed within2.3842e-7, without a bitwise-resume claim.

Affine stage commands in a separately prepared reproduction namespace, after fixed33 completion and healthy resource admission:

```sh
CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py fit --gpu 1
CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py evaluate --gpu 1
```

These are reproduction references, not instructions to rerun the preserved completed directory. Its fit stage reuses immutable coefficients on interrupted export; its evaluation refuses an existing completed result. Source and publication/resource gates remain required.

The completed fit lacked external prebinding and its exact physical GPU placement could not be established: monitor saw its short-lived Python PID on GPU0. No fit was repeated. Validation was explicitly prebound before interpreter startup by reviewed `run_validation_bound.py`, observed only on GPU1, exited0, and preserved every pinned affine artifact. Full caveat and receipts are in `RESOURCE_PLACEMENT_NOTE.md` and `GPU_PLACEMENT_REVIEW.md`.

## Physical interpretations and next boundary

Raw M is a learned representation; decorrelation is not demonstrated quantum-state orthogonality. The current PSD head already includes coherent cross-channel squares. The right-F gate is O(3)-equivariant, but signed scalar inputs do not establish electronic state-phase covariance. No full-TDDFT X/Y or MO/AO transition-density labels were recovered. Geometry-derived AO integrals are a feasible ingredient in principle, but neither physical transition density nor improved predictions follows from that fact. AO/frame conventions and symmetry/degeneracy constraints remain unresolved preflight questions.

Point-charge covariance failed a nuclear-span counterexample; direct shared-H eigendecomposition failed stable degeneracy/gradient requirements. Those routes were not trained. Available velocity labels support only a bounded consistency observation so far, not an established generalization remedy. Physical explanations remain hypotheses.

The incumbent is retained. Root's separate `../current_state/ROUND03_DECISION.md` records the passed gate and a next preparation decision for matched f-only nonlinear readouts on the two frozen prediction sources. This heartbeat performs no nonlinear implementation or new fit. Adding predicted energy would confound feature and nonlinearity changes unless a linear E/f comparator were included. A true matched from-scratch F/decorrelation factorial has not been found in the audited history; scratch and AO architecture work remain deferred. `reports/NEXT_DIRECTION_FEATURE_CLARIFICATION.md` and `reports/CALIBRATION_HISTORY_CHECK.md` preserve the feature-control and historical-novelty checks.

All lightweight objectives/settings/source/logs/results/reviews/decisions must first be downloaded to `D:\MTO\archives\`, then staged-file inspected, committed and pushed. Weights, optimizer states, raw datasets, identity/index/prediction arrays, caches and credentials remain server-only. Publication success is established by monitor's verified receipt, not by this report. The overall substantial-improvement objective remains unfinished.
