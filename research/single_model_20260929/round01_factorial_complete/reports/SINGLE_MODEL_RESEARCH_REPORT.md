# MTO single-model oscillator-strength research report

**Status: round01 complete and independently reviewed, 2026-09-30. All four arms select the unchanged starting checkpoint. No new accuracy improvement or independent-seed promotion is supported. The overall research objective remains unfinished.**

## Verified baseline tiers

Every eligible prediction below comes from one neural model. Checkpoint selection used validation. All quoted historical test scores were already exposed before this campaign; none is fresh holdout confirmation.

| Recipe | Validation pooled raw-f R² | Historical reused-test R² | Eligibility and limitation |
|---|---:|---:|---|
| Native eta0 seed11, epoch33 | .4052941183 | .4558151093 | One checkpoint; original LE+Ls-selected checkpoint. Current native anchor. |
| Same eta0 plus fixed clipped affine output calibration | .41665766 OOF; .4181192388 full-validation refit | .4596672331 | One composite checkpoint exported. Full-validation refit is optimistic; OOF does not undo prior neural validation selection. Historical calibration test exposure, worsened MAE/bright tails and an improvement interval spanning zero limit claims. |
| Frozen eta0 plus learned residual head | .4067899034, epoch2 | Not evaluated | One-model recipe candidate, but historical head/base were stored separately; not yet exported as one composite checkpoint. Only +.001496 native validation gain. |
| Historical F5, commit69bf39f | .487699605 | .508624899 | **Ineligible:** averages five model predictions. Not the baseline or final predictor. |

Native eta0 has1,552,092 parameters. The calibrated export adds two fixed FP64 buffers and no learned parameters. The residual head adds4,161 parameters. The proposed right-CG adapter adds4,016 active parameters; dormant parameters in controls are frozen and do not constitute capacity matching.

Native source checkpoint:
`/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`

SHA256: `9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136`.

Eligible calibrated single-checkpoint export:
`/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`

SHA256: `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.

The calibration is exactly `max(0, .8511830211044088*f_native + .003725185373211049)`. Returned E/A are unchanged auxiliary outputs: calibrated f **does not** obey the original E/trace(A) identity. A tensor rescaling would be another specified recipe; none was substituted here.

## Geometry-only inference and reproducibility

The calibrated checkpoint embeds all model weights, exact model configuration, training normalization and the two calibration buffers. `load_predictor` does not reopen the native checkpoint, normalization JSON or QC labels. It still needs the pinned repository model implementation and dependencies. e3nn loads its fixed O(3) library constants; those are not a trained predictor or an ensemble member.

```python
import sys
import torch
sys.path.insert(0, '/home/inspur/MTO-1/research/single_model_20260929/baselines')
from calibrated_eta0 import load_predictor

model = load_predictor(
    '/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt',
    device='cpu')
with torch.no_grad():
    output = model(z=z, pos=pos, batch=batch, n=n, edge_index=edge_index)
native_f = output['f_native']
calibrated_f = output['f']
```

Inputs are atomic numbers `z`, positions in Å, molecule indices `batch`, number of molecules `n`, and optional geometry-derived edges. No E/f/mu/A, orbital or TDDFT label is required at inference. Output shapes are E/f `(n,10)` and A `(n,10,3,3)` in original state order. Native f is `(2/3)*(E_eV/27.211386245988)*trace(A)`.

Reproduce the CPU single-checkpoint access/geometry audit:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  /home/inspur/MTO-1/research/single_model_20260929/reports/audit_inference_access.py
```

`INFERENCE_ACCESS_AUDIT.json` records PASS, the single trained-checkpoint access, embedded config/stats and finite synthetic-geometry output. This checks deployment access, not prediction accuracy. The initial overly broad load-count check also saw e3nn constants and was corrected to distinguish library constants from trained checkpoints.

Full6,686-molecule/66,860-label validation replay is already recorded in `baselines/CALIBRATED_EXPORT_VERIFICATION.json`: native R² .4052941266, calibrated refit R² .4181192453; exact fixed-calibration formula parity and no historical-test replay. To reproduce the export plus full validation parity in a new server directory, once a healthy idle GPU is available:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929
mkdir -p baseline_replay_reproduction
cp baselines/calibrated_eta0.py baselines/export_and_verify.py baseline_replay_reproduction/
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  baseline_replay_reproduction/export_and_verify.py --physical-gpu 6
```

The export script refuses an existing output checkpoint, checks GPU health/occupancy and holds the shared GPU lock. It packages the existing native weights/constants without fitting, then reloads the new single checkpoint for validation replay. Conversion needs the native checkpoint; ordinary deployed inference does not. Keep every exported weight file server-only.

### Training recipes and commands

Original eta0: MTO16 channels, query32, router/head128, DetaNet backbone; seed11; Adam AMSGrad lr.001, batch64, weight decay0, gradient clip5, FP32, AMP/TF32off, two CPU threads, LE+Ls. ReduceLROnPlateau patience50/minlr1e-6; max1000/min200, early patience150. Selected epoch33, completed236. Exact source config `experiments/qm9s_eta_Ef_20260926/configs/mto_eta0.json`, SHA256 `e1b823aa48f491b7e0eaa8cc106fa712e6ba3e3256d36e23f091af6d9c07bdf9`.

Historical trainer invocation, from an isolated reproduction of the pinned experiment directory after GPU allocation:

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python trainer.py mto_eta0
```

Do not point reproduction at the preserved completed run directory: the trainer may resume existing state. Code/data/config hashes are in the campaign frozen manifest and baseline handoff.

Current factorial continuation: identical seed11/epoch33 source; full model Adam AMSGrad reset,20epochs, fixed lr1e-5, batch64, weight decay0, clip5, no scheduler. Original LE+Ls is unchanged. All arms share source weights/order and reset runtime RNG after construction. Epoch0 is eligible; native pooled validation SSE selects checkpoints, earliest exact ties. Each arm receives the same fixed calibration only as secondary reporting.

The reviewed launcher requires exact review and publication receipts; this is a reproducible invocation template, not authorization to run another experiment:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python launch.py \
  --arms control adapter decorrelation both \
  --review INDEPENDENT_PRELAUNCH_REVIEW.json \
  --publication-receipt PATH_TO_VERIFIED_PUBLICATION_RECEIPT.json
```

The launcher verifies frozen hashes, reviewed code, archive-before-push receipt, health, occupancy and shared GPU locks, then registers monitoring. Atomic last states include model/Adam/RNG/order state after each completed epoch. A mid-epoch interruption replays that epoch; selected checkpoints are retained for transaction recovery. No prediction averaging is used.

## Fixed evaluation protocol

- Frozen120,355/6,686/6,686 train/validation/test molecules, ten valid states each. All1,337,270 labels retained; no score-driven exclusions. Printed f, including zeros, remains primary.
- Pooled R² means flattening every valid molecule-state pair before global SSE/SST, **not** averaging per-state R². Validation contains66,860 valid labels, SST158.4062371581119.
- Frozen split SHA256 `141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca`; dataset `be8fadca203429575d70642b617730592693be40858dca7189c098a671596330`; raw-label file `621dc8723fb6dc96681851a26e4f52fcb7747a4778e57a249a75d22f3af4c632`.
- Bright-tail thresholds .0549/.2412 are derived from training f and fixed. Report their SSE/MAE/RMSE, all ten per-state metrics, pooled MAE and energy errors alongside pooled R².
- Historical tail comparisons used older validation-derived cutoffs .0546/.2377. They are not directly comparable to this round's TRAIN-derived .0549/.2412 cuts. Historical calibration worsened bright-tail errors despite a higher pooled R²; exact historical figures remain in the baseline evidence rather than being relabeled under the new thresholds.
- No verified fresh holdout exists. The158 uncurated source logs have no usable excited-state labels/complete geometry, so they cannot become a fresh holdout. Pilot and continuation decisions use validation only. The legacy loader allocates shared array containers, but no test indexing, inference, scoring, fitting or selection is allowed.

## First factorial: completed null result

Frozen manifest SHA256 `eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3`; independent prelaunch review PASS. Epoch0 full validation replay R² .4052941268515, fixed-calibration .4181192441517.

| Arm | Selected epoch | Selected native / fixed-cal R² | Epoch20 native / fixed-cal R² |
|---|---:|---:|---:|
| Control, original model λ0 | 0 | .405294116 / .418119236 | .366655307 / .394554483 |
| Shared right-CG adapter F, λ0 | 0 | .405294117 / .418119236 | .366210889 / .394237328 |
| Raw-state decorrelation, λ.001 | 0 | .405294117 / .418119238 | .366738959 / .394629619 |
| Both | 0 | .405294115 / .418119236 | .366296425 / .394313638 |

All selected model tensors are bitwise identical across arms and equal the source eta0 tensors; adapter mixing is zero. Differences around1e-9 in replay R² are numerical variation. Each run completed20epochs/37,620updates normally. Independent review verified all41 frozen hashes, all21 history rows, native-SSE earliest-minimum selection, checkpoint/optimizer/RNG contents, every paired train permutation, all66,860 validation labels, and recomputed selected pooled/state/tail metrics. No experimental failure or restart was caused by the administrative agent-capacity interruption. See `INDEPENDENT_TERMINAL_INTEGRITY.json` and `INDEPENDENT_SCIENTIFIC_REVIEW.md`.

The decline is not uniform. For control, native SSE rises94.2051→100.3257 while raw-f MAE improves .0177856→.0174518, energy MAE improves .0913883→.0868585eV, and true q90/q99 tail SSE improves60.4711→57.0282 and30.9031→28.8658. State SSE improves for eight states but worsens by9.3017 for S7 and1.5589 for S8. At epoch1 the original normalized validation objective already improves while raw-f R² declines; by epoch20 trace validation error also worsens. Online training losses decrease, but these are changing-weight batch averages, not full fixed-checkpoint TRAIN accuracy.

Posthoc all-label diagnostics locate the excess SSE: control's true-dim/predicted-bright bin at TRAIN q99=.2412 adds10.0074 SSE, more than the total6.1206 increase; the other three bins improve. The largest0.1% errors account for22.83%→31.95% of SSE. All66,860 labels remain in the score. Global S7/S8 summed-error SSE worsens23.6963→41.8414, so simple anticorrelated strength redistribution does not explain the whole regression. Tight-gap cancellation remains local descriptive evidence. This auxiliary direct-pair analysis uses the rounded cutoff .0287eV; it differs on five adjacent pairs (two S7/S8 pairs) from the exact frozen quantile .028699999999999726. The reported519 S7/S8 small-gap pairs use the rounded cutoff. No primary metrics or selection depend on it.

Both interventions were active: F changed raw input by about.75–.78% on the fixed256-TRAIN audit, and the penalty reduced normalized overlap beyond control. Neither improved selection. Same-epoch20 factorial interaction is only1.8843e-6 native R²; there is no useful synergy evidence. The gap-bin audit evaluates selected epoch0 only and mixes different state/brightness compositions; it cannot explain epoch20 deterioration or establish a degeneracy mechanism. These diagnostics use reused validation and cannot establish causality or fresh generalization.

## Independent-seed confirmation: no candidate promoted

Source contracts seed23 epoch53 and seed37 epoch44 are CPU-verified. Replay expectations .3687088483/.3944553391; checkpoint hashes and exact settings are in `confirmation_preparation/SOURCE_CONTRACTS.json`. Candidate-neutral preparation and science review passed. No confirmation candidate was selected and no fit was launched by preparation.

The completed pilot does not meet the predeclared≥.003 promotion rule for control or any architecture. No seed confirmation is authorized from this result. Prepared source contracts remain available for a later eligible recipe, with matched within-seed controls and individual predictions. Independent seeds would be training replication, not fresh-data validation.

## Physical and label limitations

Current C C^T output already contains coherent cross-channel interference. Raw M is a learned representation, not an identified electronic wavefunction. Signed0e gate inputs are rotation invariant but do not guarantee electronic state-sign covariance F(-M)=-F(M). Penalty sign invariance is a separate tested property. Weak representation decorrelation is an empirical regularizer, not verified quantum-state orthogonality.

E/mu/f reconstruction agrees at printed-label precision; this does not reconstruct transition densities or establish full X/Y, MO coefficients or AO integrals. Exact/near degeneracy, phase conventions and symmetry prevent naive vector or eigendecomposition interpretations. The direct shared-H eigendecomposition candidate failed synthetic sorting/degeneracy/gradient checks. An explicit smooth factor mixer remains an untested, gauge-unidentified architecture idea.

Optional velocity/magnetic transition labels exist, but their training coverage, conventions and quality require an audit before auxiliary use. Ground-state energies/permanent dipoles/orbital gaps are not extracted fields. See `AUXILIARY_LABEL_AVAILABILITY.md`; no new QC input may be required at deployment.

A later bounded TRAIN-only audit sampled4,096 molecules/40,960states. All sampled velocity fields were present, and reconstructed velocity f matched printed f within propagated rounding intervals. Reconstruction MAE2.698e-5; velocity-versus-length strength MAE.0016802. These are label-consistency results, not prediction accuracy or proof of vector phase alignment. Full TRAIN optional-field coverage/ambiguity remains unaudited; no auxiliary fit is selected. See `velocity_audit/VELOCITY_TRAIN_AUDIT.md`.

## Root decision and next bounded question

The root decision is mirrored in `../current_state/ROUND01_DECISION.md` and `../current_state/COORDINATOR.md`. The strongest recorded eligible recipe remains the fixed calibrated eta0 composite checkpoint, with the exposure and E/A-consistency limitations above. No continuation or architecture from round01 replaces it.

Round02 is authorized for implementation and preflight only: train the existing4,016-parameter F with all original parameters and buffers frozen, comparing original LE+Ls against LE plus all-valid raw-f MSE divided by fixed TRAIN population variance, coefficient1. Keep the current1e-5/20epoch schedule, exact paired order, epoch0 eligibility and secondary fixed calibration. Verify variance provenance before source freeze. This isolates a new location before CG with coupled E/A effects; historical frozen h-only residual work already used normalized raw-f MSE and achieved only+.001496 at lr1e-4. Freezing or raw-f supervision alone is therefore not a novel rationale.

Cache raw geometry-derived M for training only after real TRAIN full-versus-cache E/A/f, F-gradient and next-Adam-update parity checks, with exact frozen-parameter/buffer checks. Final inference must reconstruct M from geometry using one composite checkpoint. Independent review, source freeze, an archive-first preparation publication and healthy resource admission must precede any new fit. No fitted loss scaling, clipping of outputs, label exclusions or test scoring is authorized.

If a later adapter-only study freezes the representation-producing model, raw-M decorrelation becomes constant and must not be presented as an active optimization factor. Any later decorrelation study must specify the trainable representation-producing subset and a matched control. This does not change the current full-model factorial.

Behind completed-factorial review and adapter diagnosis, an optional future hypothesis is phase-insensitive velocity-f or p p† auxiliary supervision in the same geometry-only model. It needs complete training-label coverage/mask/convention checks and paired controls, keeps raw length-f evaluation, and cannot assume exact length/velocity equality or infer a prediction noise floor from their label discrepancy. No X/Y/NTO reconstruction or new fit is implied.

Before closing each experimental round: download explicit lightweight records from server to `D:/MTO/archives/`, then inspect staged files, commit and push. Include settings/source/logs/results/analysis/decisions and checkpoint locations/hashes, never weights, optimizer states, raw data, prediction arrays, caches or credentials. This report is ready for the round01 archive; publication success is established separately by the publisher's verified receipt.
