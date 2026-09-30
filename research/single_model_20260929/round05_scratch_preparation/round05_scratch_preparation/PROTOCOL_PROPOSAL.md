# Proposed fresh-initialization MTO factorial on QM9S v2

Status: selected by root for preparation, 2026-09-30. Implementation and bounded TRAIN-only technical preflights are authorized after the QM9S v2 split passes independent review/freeze. The actual60-epoch fits require final source/preflight review, D-first preparation publication and a separate bound root execution decision. No new model code, statistics, checkpoint, fit, inference or split array is created by this document. Round04 is complete and published at 8497e0ba; its scalar maps will not be extended. More ambitious QC changes remain a separate follow-up and do not delay this baseline/factorial.

## Concrete question

Does learning the existing shared right-operand adapter jointly with molecular representations from initialization improve raw oscillator-strength prediction? Does the existing weak raw-state decorrelation help in that joint-learning setting? These are empirical inductive-bias questions. The features are not wavefunctions, and neither a dipole operator nor electronic-state orthogonality is identified by this experiment.

Use the upcoming frozen QM9S v2 component-disjoint 90/5/5 partition. This is a new partition of historically exposed data, not an external fresh dataset. No old checkpoint, old target statistics, fixed affine coefficients, residuals or cached representations may enter any arm. New test targets remain sealed.

## Why this is not an unchanged failed repeat

| Prior experiment | Initialization / settings | Evidence | Distinction here |
|---|---|---|---|
| Round01 control/F/decor/both | All start old eta0 seed11 epoch33; all original parameters train; LR1e-5, 20 epochs, LE+Ls, lambda=.001 | All select epoch0; F moves and decor changes Gram overlap | Current question is joint representation learning from random initialization at the original scratch LR |
| Round02 frozen F | Same old selected backbone frozen; 4016 F parameters; LR1e-5, 20 epochs, LE+Ls versus LE+raw-f MSE/variance | Tiny gains <.0002; no promotion | No frozen or validation-selected backbone |
| Historical h-only residual | 4161 head parameters on old frozen model; raw-f MSE, LR1e-4, 20 epochs | Best .406790, tiny gain then deterioration | F acts before CG and jointly changes representation learning |
| Historical seed23/37 original MTO | Original readout, 100 epochs; selected53/44 | Validation .368709/.394455; rising train scores and declining late validation | Shows need for within-seed matched controls; does not test F/decor |
| Historical scratch readout/objective study | 100 epochs; scaled objectives; MTO .373493 versus direct-f .343782 | Different readouts/objectives | Not the same shared right-F/raw-M factorial |
| Channel64 G1/G3 | G1 original tensor eta1; G3 p_outer eta1 plus E-squared trace | .405255 / .368101; different selections/budgets | Neither is a width-only control or this factorial |

The evidence does not predict that scratch F will succeed, establish a capacity ceiling, or prove a causal explanation for prior validation degradation. Exact historical source references are in history_baseline.md and the published round reports. No controlled dropout/weight-decay sweep is established by that audit; adding one now would confound this question.

## Four arms, one changed factor at a time

1. Original MTO control: F disabled, lambda=0.
2. F: shared existing 4016-parameter identity-initialized adapter, lambda=0.
3. Decorrelation: F disabled, lambda=.001.
4. Both: same F, lambda=.001.

Use the already tested mathematics in single_model_20260929/architecture/model.py, copied into a new sealed namespace with its provenance. F transforms only excited right operands Mk, k=1..10, in CG(M0,F(Mk)). M0, untransformed invariant skips and raw2e skips remain original. Scalar invariant gates and within-irrep channel mixing preserve O(3), including reflections. Zero final mixing gives exact identity and live initial mixing gradients; earlier gate weights may initially have zero gradients by construction. Controls may allocate the same dormant adapter schema but must explicitly report that they are not active capacity-matched controls. Added capacity is part of the architectural change.

Decorrelation uses the existing dimension-balanced concatenation of raw excited-state irreps, divides by the state norm with epsilon1e-8, and averages squared off-diagonal normalized overlaps over valid pairs. Exclude M0. Include low-norm states using the existing clamp; do not omit them. Apply valid masks before arithmetic. No decorrelation of transformed F(M), mu, A or batch-centered features. The value .001 remains the previous fixed conservative coefficient; inspect its train-only loss/gradient magnitude but do not tune it to v2 validation.

## Matched recipe and bounded stages

- Same original eta0 architecture: MTO channels16, query32, router/head128, ten states; all base parameters trainable. Native prediction f=2/(3*27.211386245988) * Ehat_eV * trace(Ahat).
- Original objective LE+Ls, eta0; no raw-f term, directional tensor term, auxiliary QC objective, new backbone, dropout, augmentation or calibration in this factorial.
- Recompute normalization and E_state_mean only from new TRAIN, retaining original definitions/precision conventions and documenting raw-label versus training-array rounding. Compute q90/q99 only from new TRAIN raw f, including zeros.
- Seed11 for initialization and independent data-order generator seed11. Construct/copy exactly identical base tensors to all arms; adapter allocation must not change base initialization or shuffled orders. Archive only their hashes, never tensors.
- Adam AMSGrad, LR .001, betas(.9,.999), epsilon1e-8, batch64, weight_decay0, global gradient clipping5, FP32, AMP/TF32 disabled, two CPU threads. Fixed common LR, no adaptive scheduler: this isolates the architecture under one realized schedule, rather than merely matching adaptive scheduler rules whose realized LRs might differ.
- Proposed first bounded stage: 60 epochs in every arm. This reaches beyond the historical original-MTO seed23/37 selected epochs53/44. Startup at epoch1 and resumability checks are technical checks only; do not change settings from interim scores. Save selected best and last throughout, with complete optimizer/RNG/order/state for deterministic epoch replay after interruption.
- At stage60, stop all arms if no noncontrol arm gains at least .003 pooled validation raw-f R² over the contemporaneous control's selected checkpoint. Report a bounded-budget negative/inconclusive result; do not claim that all longer scratch training is disproved.
- If at least one arm clears .003, root may approve the predeclared common continuation of all four unchanged to epoch100 after analysis/archive/publication. Compare both stage60 and stage100 policies explicitly. This is allocation, not automatic scientific confirmation. No LR/objective/lambda search or selective continuation of only the winner.
- If epoch100 confirms the margin, propose paired control/candidate scratch runs for seeds23 and37 with the same total budget and new TRAIN stats. Each remains one model/checkpoint; report individually and never average predictions. Test scoring still requires a separately frozen final choice and authorization.

Checkpoint selection: minimize full-validation pooled raw-f SSE (equivalently maximize pooled R²) among all completed epochs including initialization; earliest exact tie. All masks/labels/state order fixed. Report selected-best recipe comparisons and aligned fixed60/fixed100 comparisons separately; a best-selected factorial interaction is pipeline-level descriptive, not mechanistic synergy. The baseline itself defines a new benchmark result even if all added factors fail; historical old-split .418119 is not its matched comparator.

## Required diagnostics and preflight

1. Target-access guard: new TRAIN only for fitting/statistics; validation only in evaluator; test targets/predictions never opened. Verify split/hash/group contracts and exact new TRAIN definitions first.
2. Fresh initializer access audit: no initial_model.pt, old best/last, full-corpus normalization or learned historical values. Exact base parameter hashes agree across arms and original fresh MTO. With F identity, synthetic geometry E/A/native-f agree; selected real TRAIN parity fixture only after authorized preparation.
3. O(3) rotations/reflections and translations, atom permutations, valid-state masks including zero/one valid state, finite loss/backprop, live adapter mixing gradient, decorrelation invariance, low-norm clamp behavior. Reuse unchanged established checks; add only new initializer/loader/resume checks.
4. CPU synthetic then bounded admitted GPU TRAIN-only next-update/resume parity. Check shared base initialization and first two minibatch IDs/order before full fit. No test/validation fixture for this check.
5. Log TRAIN LE/Ls/raw-f error, raw-state overlap/norm summaries, F relative movement, total/preclip gradients and clip fraction. Audit adapter/decor gradient contributions on one fixed TRAIN subset before launch; no outcome-driven lambda change.
6. Evaluation includes every valid raw-f label, per-state SSE/R²/MAE, TRAIN-q90/q99 SSE/RMSE/MAE and all four true/predicted brightness-bin counts/SSE. No label reordering or gap-based exclusion. Export one self-contained geometry-only checkpoint and verify load/forward access; raw arrays and checkpoints remain server-only.
7. Source/preflight independent review, exact frozen settings/closure, D-first preparation archive then inspected commit/push, fresh healthy GPU UUID admission before imports, per-GPU lock and owned PID/start/boot registration precede fitting.

## Compute and decision limits

Round03's 96,284-molecule original source took about114s/epoch on A800. Scaling to about120k TRAIN suggests roughly143s/epoch before adapter/backprop and validation overhead. Budget 2.5–3h wall time per60epoch arm; four admitted GPUs give approximately10–12 GPU-hours total, with at least25% operational headroom until a real matched first-epoch measurement. A common continuation to100 adds about1.7–2h wall time/four GPUs. These are estimates, not deadlines or grounds to abandon healthy jobs.

Preferred physical GPUs1,2,4,6 only after immediate health/occupancy/UUID admission; GPUs3/7 remain excluded, GPU0 unrelated. Monitoring every four hours and resumable checkpoints persist. No result currently suggests that this bounded change alone reaches0.60.

## Separate QC follow-up

Ordinary polar-vector/rank-one heads with anonymous state slots and naively equivariant AO transition densities have point-group/electronic-character obstructions. A sign-free loss does not fix the function class. Existing even C=betaI/sqrt3+Q, A=CC^T already supports PSD tensors and coherent within-state terms. The specialist must resolve any proposed response-mixer gauge and its matched-control gradients before a separate QC-inspired pilot. A U-squared incoherent control at U=I has dead off-diagonal gradients, so it is not a fair live-optimization comparator to coherent mixing. AO integral verification and a correct phase-blind factor/covariance representation remain separate prerequisites. No such implementation or fit is authorized by this proposal.
