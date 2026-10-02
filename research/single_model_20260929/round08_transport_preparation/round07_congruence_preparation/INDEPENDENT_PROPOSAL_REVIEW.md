# Independent Round07 proposal review

**PASS for the scientific proposal only.** The original/scalar/tensor comparison asks a distinct, bounded question within the audited history. It does not authorize implementation, data access, numerical tests, optimizer updates or fits. Root decides whether to allocate preparation.

## Mathematical contract

For finite PSD inputs, epsilon>0 makes B positive definite with trace 1. Its Frobenius definition gives q in [1/3, 1/sqrt(3)]. With |b|<=.25, I+bB is invertible and positive definite, with eigenvalues bounded by .75 and 1.25. Congruence therefore preserves PSD and rank at fixed incoming A. The scalar and tensor arms coincide for isotropic R and match the Frobenius norm of the matrix perturbation. This does not equalize every output gain or expressive property.

The exact strength expansion is tr(A')=tr(A)+2b tr(BA)+b² tr(B²A). At b=0 its derivative is 2tr(BA); the scalar derivative is 2q tr(A). Both are positive for nonzero PSD A. Zeroing the last gate layer preserves the original predictor and allows a live final-layer gradient on a suitable nonzero-error fixture. Earlier gate layers initially have zero gradient by design. These statements do not promise nonzero loss gradients for zero inputs or cancelling residuals.

The transform is O(3)-covariant, including improper rotations, and inherits translation/atom-permutation invariance from the backbone and counts. It depends on A rather than an arbitrary factor C, so independent right-orthogonal factor gauges and real dipole sign changes leave it unchanged. The exact-degenerate claim is properly limited to real dipole outer products mixed orthogonally within an equal-energy block. Block sums and outside-state outputs are then unchanged; individual strengths need not be. This gives no general electronic-character or near-degenerate root-matching guarantee.

All ten untransformed predictions enter the molecular features regardless of label masks. Fixed unit scales, epsilon, element order and gate inputs are explicit. No target or unavailable QC feature enters inference. E is unchanged in the forward output, but the differentiable energy gate features transmit trace-loss gradients into the energy head after the final gate layer moves. That coupling belongs to both modified recipes.

## Controls, selection and evidence

The modified arms share the same 177 active gate parameters, features and identity initialization. The original arm's gate is dormant, so only scalar versus tensor is actively parameter matched. A separate gate RNG preserves the inherited fresh base hashes and data order. Common LE+Ls, TRAIN statistics, masks, batch/optimizer/precision/clipping and 60-epoch/112,860-update budget are fixed. Diagnostic code is shared, but intentionally different transforms and CUDA execution do not imply bitwise trajectories.

The tensor-specific gate requires at least +.003 pooled raw-f R² over each contemporaneous control and the retained .44716940136585204 reference. Validation selects earliest minimum all-label SSE, including epoch0; selected-best and fixed60 results stay separate. All 66,860 labels, zeros, per-state and true-tail/false-bright diagnostics remain required. The bootstrap is conditional on selected predictors and reused validation. TEST remains sealed, and the v2 grouping claim is limited to the audited conservative identity rules.

A better scalar or original control must be reported and may become a descriptive single-checkpoint reference. It is not evidence for tensor context and grants no automatic confirmation budget under this tensor-specific proposal. Root retains that later decision. No model or checkpoint predictions are averaged.

## Novelty and limitations

The reviewed Round01/02/05 F/decorrelation studies, scalar calibration/spline studies and Round06 raw-f objective pair did not test this matched common A-dependent Cartesian transform. This is a statement about the audited records, not a claim that molecular gates or shared operators are universally new. The original decoder already has coherent channel interactions and PSD output capacity. The new restriction adds invariant tensor-overlap context beyond a common molecular scalar rescale; it does not add external physical information.

Trace supervision does not identify physical tensor directions. The module at fixed inputs cannot revive zero A, change rank or raise one state's strength while lowering another through opposite signs of b. A trainable base can change those inputs. Full-TDDFT amplitudes, densities, NTOs, electronic phases and response operators are not reconstructed. Prior negative results, rare-error tradeoffs and the unexplained control-repeat variability caution against predicting a gain or attainment of 0.60.

The proposed future FP32/autograd conditioning, zero/identity, symmetry, factor-gauge, degenerate-block and one-checkpoint checks remain necessary. Identity-initialized original/wrapper parity must be distinguished from a transformed model's parity with its own export. The nine discarded TRAIN updates and fixed numerical tolerances are a proposed preparation budget only. No numerical check was run for this review; only documents, existing aggregate evidence and reference hashes were inspected.
