# Round04 independent protocol review

**PASS for the corrected protocol and bounded preparation scope.** This is not implementation approval, experimental evidence or production execution authority. No prediction/label/index/checkpoint array was inspected and no model, solver or inference code was executed for this protocol review.

Reviewed protocol SHA `6f10115e03b5861aeafa9b0ae0928da14d59863645549c7af1ffbd65dca1d48f` and rationale SHA `8448f14d58d9b0693b36599c6d7c296bed113b80906a6d84c72950627a755155`, plus root's ROUND04_PREPARATION_DECISION.md. The document snapshot must remain distinct from subsequent implementation/preflight work and from published Round03 records.

## Scientific question and controls

The four-coefficient fixed hinge basis is a cleaner first question than a small MLP. It retains one scalar f input, the same two frozen source prediction sets, identical rows/masks/targets and the existing affine application convention. It adds two fixed basis terms and two fitted parameters together. It tests added scalar-function flexibility/capacity; it cannot isolate an abstract nonlinear mechanism from parameter count and makes no physical response-operator claim.

For `phi=[1,f/s,(f-k90)+/s,(f-k99)+/s]`, each existing affine nests at `[beta,alpha*s,0,0]`. Common nonzero scaling does not change the full-rank function family. It improves the numerical representation without introducing source-specific feature choices. No E, state label, molecular context or unaligned hidden coordinates enter the map. An MLP would add initialization, optimizer, learning-rate/budget and possibly checkpoint-selection differences; those are unnecessary here.

The within-source spline-minus-affine contrasts address the expanded function family. Heldout-minus-in-sample spline compares two source procedures with different training size and prediction errors. Their interaction is descriptive. Training membership is not causally isolated. The four proposed contrasts and both-arm reporting preserve these distinctions.

The +.003 allocation rule over both the corresponding affine and incumbent is clear and applies to either arm. The additional +.003 heldout-over-in-sample requirement limits a transfer-specific allocation claim; it must not be mistaken for proof of source-membership causality. A single deterministic head supplies no independent-seed confirmation. Exact ties retain the incumbent. Historical validation reuse and post hoc choice of this family remain explicit.

## Objective, numerical checks and tail limits

Direct FP64 least squares on the **unclipped** output matches the historical affine fitting objective. Clamp only at application. Optimizing squared error after a clamp would change the objective: for a positive target, a negative pre-clamp prediction creates a flat region, and the resulting loss is not globally convex. The protocol avoids that optimizer issue.

Full-rank least squares guarantees non-increasing un-clipped fitting SSE when the affine basis is enlarged. It does not order the two clipped predictors' fitting SSE, since clipping may benefit each fit differently. The protocol correctly records both quantities rather than using a clipped-SSE claim as a success test.

Fixed rcond `1e-12`, rank four, condition number at most `1e8`, and stop-without-fallback behavior are prespecified. SVD avoids explicitly squaring the condition number through normal-equation inversion. These are numerical admissibility checks, not guarantees of accurate tails or generalization. Support counts and singular values must be reported without selecting new knots or dropping labels.

TRAIN target quantiles do not determine source-prediction support above the knots. The upper segment may be weakly supported or hinge columns nearly dependent. Linear extrapolation avoids polynomial growth but can still amplify errors, become nonmonotone or become negative before clamping. The clamp ensures nonnegative output, not accurate or bounded large strengths. A scalar map cannot distinguish two transitions with the same predicted f but different true strengths. Neither additional width nor these hinges automatically identifies false-bright transitions.

No ridge, monotonicity, caps, adaptive knots, alternative feature scale or fallback family is selected. Any numerical failure ends this fixed question until a separately justified protocol is recorded. Reused validation must not guide repairs to the scientific settings.

## Source and deployment contracts

The input file hashes named in the proposal match published Round03 ANALYSIS_RECEIPT metadata; this review did not open those arrays. The producer schema names, same calibration indices, all 240,710 valid rows and retained printed zeros follow the frozen Round03 contracts. The forthcoming preparation audit may read source predictions, masks and identities for fixed-design checks. Its target-free rule excludes `f_true`; the 4,546 true-zero count is inherited from the pinned producer/receipt, not a new independent target recount.

Both eventual exports contain the identical original full eta0 base/config/statistics plus their four coefficients and fixed scale/knots. They must load as one geometry-only checkpoint. The temporary source, baseline source file and caches must not be reopened during deployed loading/forward. E/A remain auxiliary and may not reconstruct mapped f. Fitted coefficients and derived exact slopes are model parameters and stay server-only; archive hashes/shapes and aggregate diagnostics, never parameter vectors or plots that encode them.

## Staging correction resolved

The draft initially mixed pre-execution preflight with fitting/export of real heads. The corrected protocol separates them:

1. **Preparation:** synthetic solver/failure checks, frozen-affine nesting, target-free calibration rank/support audit, and CPU synthetic-geometry one-file parity with hand-set coefficients. No real-target solve, real fitted head, outer-validation values or test access.
2. **Future authorized execution:** exactly two full-data target solves, then immutable coefficient receipts and fitted exports. Real first64 calibration geometry parity is a post-fit gate before the single outer-validation comparison; it is not a preparation requirement. Round03's earlier geometry replay used validation molecules and is not relabeled.

Root's expanded authority permits implementation of that preparation and a fail-closed execution interface after this protocol PASS. It does not permit production fitting. Missing distinct execution authority bound to the final source/review/publication must fail before decoding real targets or outer-validation values. No preparation helper may generate its own approval. Implementation review and meaningful preflight results remain pending.

## Recommendation

Proceed only with the narrowly authorized code and preparation checks. Keep the four-coefficient hypothesis fixed. No alternative architecture search or additional fits are needed to make this question reviewable. The incumbent remains unchanged, and this protocol contains no new accuracy result.
