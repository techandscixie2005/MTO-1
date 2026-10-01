# Round06 proposal: original PSD MTO with a raw-f objective

**Status: proposal only.** Round05 is complete and published at `f75a23c394c73262605b908f7eb280061e52901c`. This document authorizes no implementation, model inference, technical optimizer update or fit. Root must accept an exact preparation scope before any of those steps. Completed sources, checkpoints, splits and evidence remain unchanged.

## Question and minimum comparison

Does replacing trace-strength supervision with normalized printed raw-f supervision improve single-model pooled raw-f accuracy when the original PSD readout learns jointly from a fresh initialization?

| Arm | Objective | Prediction function |
|---|---|---|
| trace_control | LE + Ls | Original E and A=C Cᵀ; native f=c E tr(A) |
| raw_f | LE + Lf | Identical original E/A/native-f function |

Here `c=2/(3*27.211386245988)`, with predicted E in eV and dipole outer-product A in squared atomic units. Both arms retain all original trainable parameters, trainable energy offsets, geometry backbone, MTO coupling and PSD decoder. F is disabled and its4016 allocated checkpoint-schema parameters remain frozen in both; the raw-M penalty is zero. Reuse the proven disabled-F construction to avoid an unrelated constructor change, with bitwise initial tensor hashes matching Round05's fresh control. Neither branch inherits trained weights.

No new head, calibration, response operator, QC label, dropout, weight decay, learning-rate schedule, augmentation, selective label removal, tail reweighting or prediction averaging is introduced. Only the intensity objective changes.

## Exact targets, units and masks

For a batch, let `mE=mask_E`, `mA=mask_E & mask_A`, and `mf=mask_E & mask_f`. Use the unchanged original energy term:

`LE = mean((E_pred[mE] - E_target[mE])²) / sE2`.

The control exactly preserves the existing `base_loss` intensity term:

`Ls = mean((tr(A_pred)[mA] - tr(A_target)[mA])²) / (3*sA2)`.

The candidate is:

`f_pred[mf] = c * E_pred[mf] * tr(A_pred[mf])`

`Lf = mean((f_pred[mf] - f_printed[mf])²) / variance_f_train`.

Select valid entries before nonlinear product and squared-error arithmetic, especially the candidate E*tr(A) product. The unchanged control's base_loss may extract linear traces before valid indexing; retain that exact implementation. Do not compute nonlinear products/errors on invalid entries and mask afterward. Empty valid sets fail. The pinned v2 data contract already establishes all10 state labels valid for all assigned TRAIN/validation molecules; assert those masks remain all true and fail on disagreement rather than silently reducing the benchmark. Thus every supervised intensity term includes the same labels in this experiment. The original control mask implementation remains unchanged.

The target is the independently stored printed oscillator strength, including all22585 zero labels in new TRAIN. It is not f reconstructed from target E/A, a transformed/log target, a per-state rescaling, or teacher-forced target energy. All66860 validation labels remain included. No target or prediction clamp, cap, floor, absolute-value repair or finite-row exclusion is permitted. Mathematical nonnegativity follows from softplus E and PSD A; numerical nonfinites fail explicitly.

Both training objectives execute in FP32, as the baseline does. E/A targets retain their original FP32 representation. Printed-f labels are cast to FP32 for training; their authoritative FP64 values remain available for FP64 raw-f evaluation. The common numerical constant and normalization scalars are converted under the same FP32 arithmetic rule. Evaluation reconstructs f from predicted E and A in FP64 using the exact displayed conversion, then computes pooled all-label SSE/SST. This preserves the published raw-f primary metric while making the training-precision difference explicit.

## Fixed TRAIN-only normalization and loss weight

Reuse the exact v2 TRAIN statistics, SHA `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`:

- `sE2=.5376833706691944`.
- `sA2=.13245475393297818`.
- `variance_f_train=.0025089892829484074`, the pooled population variance (ddof0) of all1203550 valid printed TRAIN strengths, including zeros.
- Original TRAIN-only per-state energy means and `n_ref=18` remain identical.

Both LE and the selected intensity term have coefficient1. Ls uses the original tensor-based normalization; Lf uses the prespecified pooled raw-f variance, following prior raw-f studies. This makes both terms dimensionless but does **not** match their gradient norms, Hessians, tail influence or optimization speed. The change in gradient scale and E/A coupling is part of the objective contrast. Do not tune a coefficient, rescale gradients, match a validation curve, or introduce an arm-specific LR after inspecting outcomes. A future bounded TRAIN fixture may report gradient ratios without using them to choose a new weight.

## Split, initialization and common budget

Use the sealed v2 split `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155` and independent reconstruction `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`: TRAIN120355, validation6686, TEST6686. The reviewed selected-row reader must keep TEST numeric targets sealed; raw-byte hashing/ZIP transport is distinct from numerical decoding. No full target array may be decoded then sliced. Neither array creation nor target inspection is needed for this proposal.

Both arms rebuild the same random seed11 model, independently of all earlier checkpoints. Expected inherited tensor hash is `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; full disabled-F schema hash is `3c5d20463a6aa1d4e7c25ee5e57e0ceaa53c17e5a89af35d7761b58e58d661f9`. Disposable technical updates cannot seed production. Use independent NumPy PCG64 order_seed11 with all60 exact TRAIN order hashes inherited from the Round05 manifest. CUDA full training trajectories are not promised bitwise identical; record the observed repeatability limits.

Fixed common recipe: original MTO16/query32/router128/head128, Adam AMSGrad, LR.001, betas(.9,.999), eps1e-8, batch64 including the final short batch, WD0, global clip5, FP32/noAMP/noTF32,2 CPU threads. No scheduler or early stop. Exactly60 complete epochs and1881 batches/epoch =112860 optimizer updates per arm. This includes the prior control's selected epoch45 and avoids an undertrained tiny pilot. The two arms are contemporaneous; the old Round05 control is an additional reference, not the only matched control.

Resource estimate from Round05: about2.7h wall time for two concurrent healthy GPUs, approximately5.4GPU-hours plus25% operational headroom. The proposal prefers GPUs1/2 under the existing UUID/health/occupancy/shared-lock/owned-registration policy, subject to fresh admission. No fallback or intervention in unrelated jobs. Persistent four-hour monitoring remains unchanged.

## Selection, reporting and allocation gate

Evaluate the entire validation partition at epoch0 and every completed epoch. Select each model by the earliest minimum pooled raw-f SSE, equivalently maximum R² on fixed labels. Keep per-state errors, energy error and TRAIN-fixed true q90/q99 tails (.0549/.2406), false-bright2×2 counts/SSE, prediction/error quantiles and large-error SSE shares as diagnostics. They do not retune the objective or select a different checkpoint. Report selected-best comparisons and aligned epoch60 separately.

Primary allocation gate: candidate selected R² must exceed **both** the contemporaneous selected trace control and the retained v2 reference `.44716940136585204` by at least `.003`. Thus beating an unusually weak rerun alone does not qualify. No automatic extension to100 or loss/LR sweep. If the gate fails, close this fixed pair and publish the negative evidence.

A qualifying candidate permits only a later root decision on paired fresh independent training seeds23/37 under unchanged settings; each predictor remains one checkpoint and predictions are never averaged. Seed confirmation is not fresh data confirmation. Review energy/state/tail tradeoffs before allocating further compute, without retroactively changing this numerical gate. A stronger contemporaneous trace control may become a new descriptive v2 reference, but its improvement is same-recipe execution variability, not evidence for the raw-f objective. It grants no automatic seed allocation or objective claim.

If uncertainty is reported, use paired validation-component bootstrap with a declared seed/draw count and recomputed pooled SST; state that it is conditional on validation-selected checkpoints and does not measure training-seed variation or correct adaptive validation reuse. Neither historical test nor new TEST scoring is part of this proposal. A later locked test evaluation requires separate root authorization and one final selected recipe.

## Coupled gradients and numerical risks

Writing `s=tr(A)` and residual `r=c E s-f_true`, a valid batch of N labels gives:

`dLf/dE = 2*c*s*r/(N*variance_f_train)`

`dLf/dA = [2*c*E*r/(N*variance_f_train)] I`.

Both paths backpropagate through their shared representation; no detach, target E substitution or frozen energy branch. LE anchors energy, but the product still admits strength/energy compensation. For A=C Cᵀ, the corresponding direct factor derivative is proportional to C; near-zero C can have small gradients, and unusually large strengths can produce large residual gradients. The trace objective also has a quadratic-factor path, so this is not a new claim of unique instability. Fixed global clip5 and finite checks apply identically. A nonfinite failure is preserved and investigated; do not silently cap predictions, drop rows, lower LR, raise tolerances or restart.

The proposed engineering preflight, only after a new preparation decision, should compare the analytical gradients with autograd on synthetic PSD matrices (including zeros, small E/C, bright strengths and invalid-mask sentinels), verify exact unchanged control loss, and demonstrate live gradients through both original E and A branches. Include proper/improper O(3), atom permutation/translation, one-checkpoint access/parity and finite mixed-batch reductions. Reuse the reviewed source/split boundary and explicit post-diagnostic CUDA tolerance policy; any newly relevant acceptance criterion must be fixed before execution.

For a concrete later bounded TRAIN check, propose the same first128 TRAIN rows/two64 batches, three updates per arm (update1, update2, restored replay of update2): **six discarded updates total**. Report objective/branch gradient norms and clip status without adjustment; verify model/optimizer replay within the already justified fixed tolerance and exact RNG/order. No validation or TEST fixture, no production initialization from technical state. This preflight is proposed, not authorized by this document.

## Historical nonrepetition and physical interpretation

The audited history has no matched fresh original-PSD LE+Lf versus LE+Ls pair. Historical R².373493 was MTO-direct-f with beta/tensor heads removed, different f/energy initialization and a100-epoch schedule; it was not this PSD function. Earlier original-PSD raw-f/E²-trace full-model trials were LR1e-4 continuations of a selected eta0 checkpoint and chose epoch0. Round02 froze that base and trained only F, with a tiny raw-f gain that failed its gate. Those are negative evidence, not permission to claim this pair will succeed. Source-backed distinctions are in `round05_scratch_preparation/HISTORICAL_NONREPETITION.md` and `history_baseline.md`.

Round05 now rules against unchanged fresh F/decor ablations under its tested single-seed60-epoch recipe. Its TRAIN errors decrease after validation-selected epochs, while validation trace/raw-f errors worsen; F is active and decorrelation changes the intended feature statistic. This motivates testing the objective directly but does not prove objective mismatch caused the errors. Rare-error concentration and seed variation remain plausible limits; squared raw-f loss can still overfit.

The oscillator-strength relation is the existing length-gauge observable formula. Replacing its loss does not reconstruct transition amplitudes/densities, NTOs, orbital populations, TDDFT X/Y or an electron–hole response operator. Trace supervision is phase-blind and does not identify physical tensor orientation. Under exact degeneracy only suitable block sums have invariant physical meaning; the benchmark's ordered state labels remain unchanged. No root matching, state swapping or exclusions are allowed. Geometry-only inference requires no QC labels, and this objective does not introduce a new AO, symmetry-frame or origin convention.

The deferred177-parameter common PSD congruence would additionally test dependence on latent cross-state tensor orientations and needs its matched scalar-gate control. Keep it separate. The simpler two-arm objective question changes fewer assumptions and costs less; neither is a prediction of attaining R²0.60.

## Recovery, artifacts and required next decision

Reuse reviewed atomic completed-epoch last.pt recovery with model/optimizer/Python/NumPy/CPU/CUDA RNG/order/history and immutable selected-version hashes, separate best.pt and geometry_best.pt, explicit failure evidence, CPU-safe optimizer restoration and completed-run refusal. Export one complete original geometry predictor with embedded config/TRAIN stats and strict buffers. No ensemble or averaging.

Before implementation or the six technical updates, root must accept a bounded preparation decision. After reviewed source/preflight, FIRST download lightweight records to D:\MTO\archives, inspect/commit/push, then require a separate exact execution authorization for two60-epoch fits. No action is implied by proposal review or publication alone. Keep all checkpoints, optimizer tensors, raw/prediction/split/identity arrays and caches server-side. Root decides remaining scope; there is no unresolved routine setting that requires a search over options.
