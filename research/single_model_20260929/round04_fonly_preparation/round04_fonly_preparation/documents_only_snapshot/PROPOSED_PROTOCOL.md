# Proposed Round04: a shared scalar spline on frozen oscillator strengths

**Status: documents-only protocol snapshot, pending final independent review.** Work that produced this snapshot included no implementation, fit, inference, prediction-array inspection or validation scoring. Round03 is complete and published at commit `4f9ae50647f957ac7d68ab1a20200a79fa527cf9`. Its records remain unchanged. Root has accepted the proposed function, solver/failure rules and promotion thresholds subject to final independent review. Any expanded preparation authority must be recorded separately; production execution requires source freeze, preflight, independent review, archive-first publication and a separate execution decision.

## Question and fixed comparisons

Does a small increase in the flexibility of an f-only correction improve raw-f accuracy beyond the corresponding frozen affine map? Does that increment differ when the correction is learned from the clean source's held-out predictions versus the full baseline's in-sample predictions?

Fit exactly two maps on the same original-TRAIN calibration rows:

| Arm | Scalar fitting input | Frozen same-input affine control | Deployed backbone |
|---|---|---|---|
| Held-out spline | Round03 clean source's native f | Round03 held-out-source affine | Original full eta0, epoch 33 |
| In-sample spline | Round03 full eta0 native f | Round03 in-sample affine | The same original full eta0, epoch 33 |

Both final maps act directly on full eta0's native f. Keep the native backbone and the historical validation-fitted affine as additional fixed comparators. No map consumes another map's output. There is one shared scalar function across all ten states and molecules; no state index, predicted energy, other context, hidden feature, source-model feature alignment or QC input enters it.

## Function: four fitted coefficients, two fixed knots

Let `s = 0.050109692115345626`, the already audited population standard deviation of every valid raw-f label in original TRAIN. Let `k90 = 0.0549` and `k99 = 0.2412`, the TRAIN thresholds frozen before Round01. Given scalar native prediction `f`, define:

```text
u = f / s
phi(f) = [1, u, max(0, (f-k90)/s), max(0, (f-k99)/s)]
q(f) = b + w*u + c90*max(0, (f-k90)/s) + c99*max(0, (f-k99)/s)
f_out = max(0, q(f))
```

The four fitted scalars are `(b, w, c90, c99)`. The scale and knots are fixed buffers, identical for both arms. Every original backbone parameter and buffer remains frozen. The affine control nests exactly at `(beta, alpha*s, 0, 0)` for each source. This anchor is retained as a separate fixed predictor, not as an initializer that must be optimized.

The map is continuous, with three linear segments before the existing output clamp. Its slopes are `w/s`, `(w+c90)/s` and `(w+c90+c99)/s`. Extrapolation above the largest knot is linear. Slopes are unconstrained; negative slopes and zero outputs are possible and must be reported. There is no monotonicity constraint, ridge penalty, learned knot, output cap or adaptive support rule.

The knots locate two previously documented strength regimes, using thresholds defined before the current proposal. They are not quantiles of the fitting predictions, so their actual source-input support is unknown until the authorized preflight. Choosing this hypothesis after examining Round01–03 errors is post hoc; the old thresholds do not make this an independent hypothesis test.

## Fitting rule, initialization and budget

Use one deterministic FP64 least-squares solve per arm:

```text
theta = argmin_theta mean((phi(f_source) @ theta - f_printed)**2)
```

This is raw printed-f squared error for the **unclipped** linear-basis output. Apply the nonnegative clamp only when reporting or deploying predictions, exactly as in the existing affine fitting rule. Report both the fitted unclipped objective and the deployed clipped raw-f errors. Affine nesting guarantees no greater unclipped fitting SSE for the full-rank least-squares spline, up to numerical tolerance. It does not guarantee an ordering of the two clipped predictors' fitting SSE. Do not describe the solve as minimizing the clipped objective; that would be a different optimization problem.

The recommended direct solve replaces the initially considered MLP/SGD workflow. Therefore:

- Optimizer: deterministic FP64 SVD least squares, `rcond=1e-12`; no normal-equation inversion.
- Budget: exactly two four-coefficient solves; one per frozen source. No epochs, gradient updates, learning rate, scheduler, early stopping, restarts or hyperparameter search.
- Initialization and live gradients: not applicable to the solver. Exact affine nesting, basis support, rank and finite-solution checks replace iterative-training initialization tests. A loss gradient can legitimately be zero; it is not a success criterion here.
- Order: existing calibration molecule order, then states 1–10; keep each valid label exactly once. No random shuffle or batch subsampling.
- Seed: none for deterministic fitting. If uncertainty intervals are reported, use paired molecule bootstrap seed 20260930 and 2000 draws after coefficients are frozen; this does not change any predictor.
- Numerical environment: pinned existing Python/NumPy, FP64, two CPU threads. No GPU or model inference is needed for the coefficient solve.

Stop both-arm execution without fallback if any fitting value is nonfinite, the exact row/mask/hash contract fails, the design rank is below four, or `sigma_max/sigma_min > 1e8`. Record singular values and support counts in all cases. Do not drop rows, relocate knots, add ridge, lower rank, change precision or substitute an MLP after observing a failure. A revised scientific question would need a new protocol.

## Data, freezing and single-checkpoint export

The only coefficient-fitting input is Round03's existing server-only `affine/calibration_predictions.npz`, SHA256 `defe63f703b405b09af84a5fe5c05c742fc645385a5311e708f2bd6611c5c675`. Its published producer names the required fields `source_native_f`, `full_baseline_native_f`, `f_true`, `mask_f`, `indices` and `ids`. Energy fields in that file must not be used. The contract is 24,071 identical molecules, 240,710 valid labels and 4546 printed zeros, with calibration index hash `9547ca89feff3301f5e4c6fcd736be7a9186878a9fcb1fec8695c5c12d60707b`. Invalid entries remain masked by the frozen source mask; valid zeros remain included.

Before either map sees outer-validation inputs or labels, atomically freeze both coefficient vectors in server-only tensor files and hash their source arrays, row/mask contracts, scale/knots, solver versions/settings, full-base checkpoint and all code/configuration. Lightweight receipts contain tensor file hashes/shapes/dtypes, not fitted coefficient values. Freeze the two source-specific affine controls from Round03 without refitting them. Preserve an immutable coefficient receipt if an export is interrupted; resume export from that receipt without solving again.

Each deployment checkpoint must contain the complete original eta0 tensors, original config/statistics and nonpersistent-buffer provenance, plus four coefficient buffers and the fixed scale/knots. It must strictly load from one file and predict from molecular geometry alone. Neither clean-source weights, original baseline files, fitting arrays nor caches may be read during inference. Both checkpoints contain the same base tensor hash `be433e432c79276495f542b3012891c8954e84e8aba0b6e1b14c3299096f4bef`; only their four fitted coefficients differ. Preserve auxiliary native f, E and A for audit; mapped f can be inconsistent with unchanged E and A. No physical transition-density claim follows.

## Evaluation, selection and decision rules

There is one fixed endpoint per arm, not an epoch/checkpoint search. Freeze both endpoints before a single prespecified validation comparison. Reuse the already saved full-baseline native validation array, SHA256 `1c4d4d8b44d19a57a2bd46ccfd0f8b054564ca1cc628a65d73a4629dc8c5e92a`; no full-model validation forward is needed to compute the maps. Retain all 66,860 valid raw-f labels, existing masks, state order and TRAIN q90/q99 cutoffs. Never access test data.

Report both splines, both frozen affine controls, native eta0 and the historical calibrated incumbent. For every predictor report pooled R²/SSE/RMSE/MAE, all ten states, true q90/q99 errors, and the exhaustive true/predicted-q99 four-bin counts/SSE. Report source-fitting support in the three knot intervals, source-input extrema, zero-clamp frequency, extrapolation counts and whether all segment slopes are nonnegative. Keep exact fitted coefficients and slopes server-only; they are learned parameters. These are diagnostics only; no excluded rows or altered primary metric follow from them.

Prespecified contrasts:

1. Each spline minus its own source-matched affine: the increment from the larger fixed function family under the same fitting objective and rows.
2. Held-out spline minus in-sample spline: the empirical transfer comparison.
3. The difference between the two within-source increments: a descriptive interaction. It does not isolate training membership or establish physical causation.
4. Each spline minus the strongest historical calibrated incumbent, whose outer-validation exposure remains explicit.

**Promotion rule accepted by root, subject to final protocol review:** either arm may earn a separately designed confirmation study only if its pooled validation R² exceeds both its own affine control and the recorded incumbent by at least 0.003. Record all state/tail tradeoffs before any decision. A transfer-specific claim additionally requires held-out spline to exceed in-sample spline by at least 0.003; this extra condition does not prevent an in-sample recipe from being a candidate. Smaller gains or failures do not trigger automatic budget extensions or another function search. No independent-seed claim follows from a deterministic head fitted on this single pair of source models.

If a deployable candidate is retained, choose it only by pooled validation raw-f SSE among the frozen endpoints and existing anchors; retain the incumbent for any exact tie. Every endpoint is still reported. Validation reuse and family selection must be disclosed. Bootstrap intervals are descriptive uncertainty on reused validation, not fresh confirmation or a correction for previous model selection.

## Meaningful preflight and post-fit gates

### Preparation before a production execution decision

1. Verify published manifests, original checkpoint/array hashes and identical row/mask contracts without consulting outer-validation values. No new source training or prediction generation is needed. Prebind `CUDA_VISIBLE_DEVICES` to the empty string before starting the preparation interpreter.
2. On synthetic scalar fixtures covering both sides of both knots, verify continuity, linear tails, nonnegative clamp, known four-coefficient recovery and exact nesting of each affine formula. Verify NaN, rank-deficient and conditioning failures are rejected without fallback. Preserve the initial failures if a test needs repair. Synthetic least-squares fixtures are not production fits and must be clearly labeled.
3. Only after explicit preparation authorization, read the complete calibration design inputs/masks/identities and audit finite values, rank, singular values and source-input support. Do not solve against real targets, inspect target residuals or tune the model from these diagnostics. Raw-f objective arithmetic is tested on synthetic fixtures. Do not decode the calibration `f_true` field in this preparation audit.
4. Test one-file CPU export/access parity using synthetic geometries and hand-set coefficients or the existing affine anchors. Compare complete base tensors and all named buffers before/after; no base optimizer exists. Verify NumPy/Torch scalar-map parity in FP64 (`atol=1e-12`, `rtol=1e-12`). No cache, label or auxiliary source may be opened within `load_predictor` or `forward`. Do not use real fitted four-coefficient heads in preflight.

These preparation checks must finish before source freeze, independent review and the archive/publication gate. They do not consume either of the two production solves.

### Production only after separate execution authorization

Exactly two actual full-data target solves occur once, after all execution gates: one per frozen source. Freeze their server-only coefficient tensors and lightweight hash receipts. Verify unchanged full-base parameters/buffers and both fitted exports before any outer-validation access. An interrupted export resumes the frozen tensors without another solve.

Then verify the two fitted exports on the first 64 calibration molecules in frozen order, with no label-based selection. This is a new fixed fixture; Round03's original geometry replay used its first 64 validation molecules. For native f, retain Round03's saved-prediction tolerance (`atol=2e-6`, `rtol=1e-5`). Separately verify each scalar map against FP64 arithmetic on the identical recomputed native f (`atol=1e-12`, `rtol=1e-12`), so slope amplification of base-rounding differences is not mistaken for scalar-map implementation error. Record differences without changing tolerances. The post-fit fixture is not part of current preparation authority.

Before reporting the one validation comparison, verify coefficient-freeze chronology, native/historical control metrics against Round03 within `1e-7` pooled R², all counts and exhaustive-bin sums. A mismatch stops reporting without coefficient changes or repeated model scoring.

## Cost, authority and archive

The two design matrices have 240,710 rows and four columns each, about 7.7 MB per FP64 matrix. Two SVD solves and cached validation mapping should require seconds to roughly one minute of CPU work; this is an estimate, not a measurement. Loading two composite exports and a small geometry fixture may take a few additional CPU minutes. Budget at most ten minutes of CPU preflight/execution initially; exceeding it triggers a progress review, not an algorithm change. No GPU allocation is needed for the proposed statistical fit. Any later optional GPU parity check needs explicit allocation, UUID binding before interpreter startup and a shared lock; it is outside this proposal's current authorization.

This snapshot records documents-only work. Root has accepted the fixed-basis solver and promotion rule subject to final independent review. Any subsequently authorized implementation/preflight phase must record that expanded scope separately; production remains a separate execution gate. After an authorized round, FIRST download lightweight code/settings/logs/receipts/aggregates/reviews/decisions to `D:\MTO\archives\`, then inspect staged files, commit/push and verify the remote. All fitted head coefficient arrays/tensors, exact learned slopes, checkpoints, raw labels, indices, predictions and caches stay server-only. Do not serialize fitted coefficient values into an archived JSON receipt, table or plot.
