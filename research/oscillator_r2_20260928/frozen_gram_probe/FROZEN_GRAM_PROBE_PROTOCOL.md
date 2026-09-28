# Frozen invariant Gram probe: one fixed convex comparison

Date: 2026-09-29. Status: protocol prepared; no full-data fit or new validation inference authorized yet. Root accepted this one comparison for implementation. Execution requires independent code review, preflight PASS and root authorization. No sealed source changes, test access, ridge/feature search or new backbone training.

## Purpose and decisive comparison

Test whether tensor invariants omitted from h support a useful correction beyond h plus the model's own intensity, while holding the original eta0 representation fixed and removing iterative head-optimization and epoch-selection uncertainty. Fit exactly TWO linear residual regressors with the same train-only normalization, fixed ridge and nonnegative output projection:

- A, amplitude-aware control: h_a[128] plus native64 eta0 f_base,a.129 slopes plus one intercept,130 coefficients before constant-column removal.
- B, Gram-augmented: the same129 columns plus528 symmetric Gram coordinates.657 slopes plus one intercept,658 coefficients before constant-column removal.

Rows are individual molecule/state pairs, with all ten ordered states retained and equal SSE weight. h_a is already state-conditioned by the frozen MTO query/routing/CG path. Both arms share coefficients across states; add no one-hot state inputs, new queries, state-specific coefficients, true-energy inputs or cross-state mixing. Native base_f supplies predicted-energy coupling and magnitude, never oracle energy.

The primary feature comparison is B minus A. Also compare each once against frozen eta0 and the already-frozen equal-three ensemble, explicitly labeling single-versus-ensemble cost. Feature count/effective degrees of freedom differ; this tests adding invariant features and their coefficients, not a capacity-matched information-only theorem. Both families contain eta0 through zero correction. A linear probe cannot test every nonlinear scalar invariant readout.

## Fixed sources and Gram definition

Source eta0 epoch33 checkpoint SHA2569f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136. Use exactly the frozen base from the sealed retained_residual initial checkpoint, with the entire model eval/frozen and no residual-head correction. Reuse frozen_residual/runtime/cache_train.npz and cache_val.npz for h, native64 base_f, IDs/indices and raw targets. Their hashes are948f14528a222613cc3a590d3d8e710586374be28dc0826d43afde461de48b3b andf8f47fabcb00b0c03a99fa3d684a74294ff74fb0e9087f2301953802a6263848. Preserve all existing cache/protocol files.

The original CG output provides T[n,a,k,m] with32 tensor channels k and five l=2 components m. Use source decoder.cartesian_basis B[m,i,j], shape5x3x3. Promote frozen FP32 T and B to FP64 BEFORE forming:

    V[k,i,j] = sum_m T[k,m] B[m,i,j]
    G[k,l]   = sum_i,j V[k,i,j] V[l,i,j]

This avoids assuming that the stored finite-precision Cartesian basis is exactly orthonormal. Flatten the upper triangle in lexicographic order(k=0..31, l=k..31). Store G[k,k] for diagonal and sqrt(2)*G[k,l] for off-diagonal entries, preserving the symmetric-matrix Frobenius inner product. The resulting528 coordinates are FP64. Append them after the common h/base columns, with a saved explicit index map. No Gram clipping, log transform, PCA, random projection or feature selection is included.

G contains every pairwise contraction among these32 tensors WITHIN the requested state. In exact arithmetic, the current trace is beta(h)^2 + g(h)^T G g(h)/32, where g=tanh(tensor_gate(h)); E then supplies native f. Verify this relation against the source path within declared numerical tolerance. This is sufficient quadratic-magnitude information for the existing readout, not all possible molecular invariants or all cross-state tensor relations. A linear residual in[h,base,G] does not introduce arbitrary nonlinear h-by-G interactions.

## Cache, provenance and phase ordering

First generate TRAIN Gram features only. Use frozen input geometry/graph cache and source feature code in fixed molecule batches64; match rows to existing h/base cache by exact IDs/global indices/state order. Assert newly computed h/E/base matches existing cached values within the accepted FP32 tolerance, before using new G. Persistent FP64 Gram memmaps are practical and avoid repeated source passes: shape[120355,10,528] train and[6686,10,528] validation, with an ID/index sidecar and source/protocol/feature/dtype/hash manifest. Arrays and coefficients remain server-only.

Do not generate or inspect validation Gram features/labels to choose normalization, ridge, features or solver. Freeze both fitted coefficient vectors, preprocessing statistics, active columns, solver diagnostics and file hashes in FIT_FROZEN.json before validation Gram generation/prediction. Existing validation h/base cache is already available; its existence supplies no tuning permission. A preflight may use synthetic cases and a fixed small TRAIN batch; validation metadata may be checked without fitting or using its targets. No full-data fit during preflight.

After that freeze, generate validation Gram once and evaluate both fixed models in the same pass. Native eta0 and fixed equal-three prediction arrays are comparison outputs only. No validation results can trigger a second ridge fit, alternate feature subset, alternate projection or changed coefficients.

## Exact target, normalization and optimization

Training target is residual r=(raw_f-f_base_native64)/f_std_train, where f_std_train=.050109692115345626. Use raw source f in FP64, not log f, A-derived f, validation-fit calibration targets or a teacher residual from a trained new head.

Across allN=1203550 valid training molecule/state rows, compute FP64 population feature means mu_j, standard deviations sigma_j, residual mean mu_r and centered covariance/cross-covariance. Use stable blockwise centered accumulation (Chan/Welford merge or an algebraically equivalent tested two-pass method), not cancellation-prone subtraction of large raw moments. Assert the frozen masks are all true; missing entries would require a new protocol rather than silently changing weighting.

Remove only columns with EXACTLY zero training centered variance. Record their indices and values; never remove columns using validation. All remaining columns use x_j=(phi_j-mu_j)/sigma_j. No per-state standardization and no fitted full-dataset statistics. Common129 feature statistics must be identical for A and B; obtain A from the principal block of the same full training moment matrix.

For each arm solve the single fixed objective:

    J(b,w) = mean_train[(r-b-x*w)^2] + 0.001*sum_j w_j^2

The intercept is unpenalized. Because x is centered, b=mu_r. With C=mean(x^T x) and c=mean(x*(r-mu_r)), solve(C+0.001 I)w=c in FP64 using Cholesky. Lambda0.001 applies to MEAN squared residual error in standardized units, not summed error. No lambda grid, early stopping, cross-validation, molecule subset choice or validation-based selection. The regularization is a fixed diagnostic convention, not claimed optimal.

Require finite means/scales/moments, symmetric C, successful SPD factorization, finite coefficients and relative normal-equation residual ||(C+lambda I)w-c||2/max(||c||2,1e-12)<=1e-10. Report matrix symmetry error, minimum/maximum eigenvalues or a condition estimate, coefficient norm and effective degrees of freedom trace(C(C+lambda I)^-1). Zero c has the exact w=0 solution. Stop on numerical failure; do not change lambda or silently add extra jitter. An alternative same-objective solver requires review, not result-driven tuning.

Verify J_B<=J_A within FP64 tolerance, since the common-feature solution padded with zeros is feasible for B. Verify each solution's regularized objective<=the zero-correction baseline objective. Raw SSE need not be strictly nested under ridge, so do not assert an unjustified B-train-SSE ordering.

Prediction for BOTH arms:

    signed_f = f_base_native64 + f_std_train*(mu_r+x*w)
    emitted_f = max(0, signed_f)

Fit the convex SIGNED residual problem above; the shared nonnegative projection is a prespecified post-fit rule. For nonnegative raw truth it cannot increase squared error versus signed predictions. Report signed and projected SSE, negative count and projection's SSE effect, and verify that inequality. Do not call this the exact optimum of the projected/nonnegative training objective. This differs from the prior MLP's abs rule; causal comparison is between A and B here, not an isolated activation claim versus the prior MLP.

## Evaluation and locked comparisons

After FIT_FROZEN, report full checkpoint-fixed training and validation raw native-f pooled float64 SSE/SST/R2/MAE/RMSE, every state, true-f q90=.0546/q99=.2377 and complements, mean signed prediction error, error quantiles, signed/negative counts, maximum predicted intensity, coefficient/feature condition diagnostics, and a molecule-level error-concentration summary. E remains the unchanged source prediction and must reproduce its metrics. No state deletion, target-dependent inputs, intensity clipping ceiling or altered benchmark.

Save both full validation arrays server-only with IDs/indices/raw truth, frozen base, signed prediction, emitted prediction and residual. Save train metrics without exporting full train predictions unless needed remotely for audit. Molecule14562 is a fixed illustrative diagnostic from prior work, never a fitting weight, feature or selection criterion.

Use2000 paired bootstrap replicates, seed20260928, resampling whole molecules and whole conservative connectivity groups with all ten states and all molecules in each group retained; recompute pooled SST each replicate. Report B-A, A-eta0 and B-eta0 as primary scientific contrasts, plus fixed equal-three context for each. No checkpoint selection occurs within either ridge arm. These intervals still condition on a previously reused validation split and a feature design informed by earlier research; they are not untouched confirmation.

A practical candidate requires>=.01 R2 improvement versus eta0 and positive delta in>=80% of molecule bootstrap replicates, with connectivity robustness reported. Additional Grams are specifically useful if B beats A materially (prespecified+.005 R2 exploratory feature-value threshold); report both effect size and uncertainty, never positive frequency alone. Any nomination is for later confirmation, not test access or automatic model replacement. A wins/B does not: amplitude-aware linear correction may suffice. B winsA: extra tensor invariants are useful under this model/regularization. Neither wins: no useful linear correction under these fixed settings; this does not establish an invariant-readout or backbone ceiling.

## Resource and independent-review requirements

FP64 Gram payload is5,083,795,200 bytes train and282,416,640 bytes validation (about5.37GB total decimal). Full657x657 FP64 covariance is3,453,192 bytes. Stream memmaps in bounded blocks (suggest2048 molecules or less) and retain only moments plus existing h/base cache; do not materialize anN-by657 FP64 design matrix. Use at most two CPU threads. GPU may accelerate frozen feature generation and FP64 moment products; Cholesky is only at most657 dimensions. Record actual CPU/GPU/IO times and peak memory separately. Prior source feature passes took48.1s train and3.6s validation; Gram computation, covariance and disk writes add unmeasured time. This is an estimate, not a promise. A one-hour diagnostic budget excluding resource wait is adequate as an admission target; an overrun should be reported, not used to truncate/retune the fit silently.

Before root authorizes full execution, independently review implementation and preflight: source/cache/ID hashes; exact feature dimensions/order; Gram rotation/permutation invariance and PSD/symmetry checks on fixed train examples; trace reconstruction; synthetic centered-moment and ridge closed-form identities; streamed versus dense solution on a fixed train subset; common-feature principal-block identity; no target/oracle leakage; projection SSE inequality; masks/dtypes/finiteness; fit-before-validation phase gate and atomic resumable feature preparation. No optimizer epochs or architecture search are included.

Use a healthy idle admitted GPU1/4/6 under the existing shared lock; GPU5 only with explicit existing guarded microcheck/ECC admission. Preserve unrelated GPU0, occupied loss-pilot GPU2 and excluded faulty3/7. Check health and foreign-process occupancy through feature/moment passes. Failed health/numerical/provenance gates invalidate or pause the diagnostic; do not reset/preempt devices. Source files, dataset splits, prior selected models and all sealed studies remain unchanged.