# Why start with a fixed scalar basis

## Recommendation

Use the proposed four-coefficient continuous piecewise-linear map before considering an MLP. It is the smallest clear comparison to the two already frozen affine maps: the same scalar input, same rows, same raw-f least-squares objective and same final nonnegative clamp, with two additional fixed basis terms. It introduces function flexibility and two parameters together; it does not isolate an abstract effect of nonlinearity from parameter count.

| Issue | Fixed four-coefficient basis | Small MLP |
|---|---|---|
| Comparison with affine | Affine is exactly nested; same direct solver | Adds architecture, nonlinear optimization and often checkpoint selection |
| Initialization | No initialization; affine anchor remains explicit | Must ensure live gradients and control random initialization |
| Optimization | One deterministic FP64 least-squares solve per source | Budget, learning rate and stochastic training can affect outcome |
| Tail behavior | Linear beyond the upper knot; slopes can still be harmful | Extrapolation depends on activation/weights; more parameters do not resolve rare support |
| Support audit | Four-column rank and singular values are explicit | Effective identifiability is harder to summarize |
| Compute | Saved scalar arrays, CPU only | Also cheap if scalar-only, but adds avoidable training choices |

Neither function can distinguish a true bright transition from a false bright prediction when they have the same predicted f. A scalar map estimates an average correction; it lacks molecular/state context. It can trade dim-target errors against true-bright errors, as Round03 already showed. This limitation is structural and is not solved by increasing the scalar network's width.

## Why these knots

The only two knots are the exact original-TRAIN q90/q99 thresholds used throughout the campaign: 0.0549 and 0.2412. They separate the broad strength region from bright and very bright regions in the existing reports, with no new label/prediction scan, quantile estimation or search. One extra linear segment at each threshold permits shrinkage to differ across those regimes. Linear extrapolation avoids the quadratic growth of a polynomial correction, but does not guarantee safe or accurate high-f predictions.

The thresholds describe true TRAIN labels; they do not guarantee enough source-prediction mass above each knot. The design may therefore be poorly supported, especially in the upper segment. That is why rank, conditioning, input counts and extrema are mandatory future preflight outputs, and numerical failure stops the protocol without changing the knots. Rare-error sensitivity remains because every valid raw-f residual retains ordinary squared-error weight.

The two-knot hypothesis follows inspection of previous validation errors. Its fixed historical thresholds limit new degrees of freedom but do not erase post hoc choice or outer-validation exposure. It is a controlled exploratory comparison, not a fresh confirmation.

## What the existing evidence supports

- Round01 full-model continuation of right-F/decorrelation decreased validation accuracy despite measurable changes to the adapter and raw-state correlations. Its high-state false-bright errors grew. This does not prove that F cannot work when trained from initialization.
- Round02 frozen pre-CG F had active parameter/output changes and improving fitting objectives, but produced only tiny early validation gains. It did not meet the 0.003 threshold. Repeating the same budget or merely extending it lacks a new rationale.
- The historical frozen hidden-h residual also optimized raw-f error and gave a small early gain. The current proposal changes the fitting representation to observable f and, for one arm, uses predictions from a source that never trained on the calibration molecules. It does not transfer hidden coordinates between independently trained models.
- Round03 held-out affine reached 0.416374515 versus 0.402024351 for the in-sample affine and 0.405294124 for native eta0. That supports investigating this transfer procedure. It still trails the 0.418119244 historical calibrated incumbent, and source size/quality/optimization differ. Training membership is not an isolated causal explanation.
- A matched scratch right-F/raw-M-decor factorial is genuinely absent from the audited history, but it is a separate, more expensive question. It remains deferred rather than being folded into this scalar-map experiment.

## Decisions for root

1. Root has selected direct fixed-basis least squares in place of the originally suggested iterative nonlinear head, subject to final independent review. Initialization, live-gradient and learning-rate requirements become inapplicable; exact affine nesting, rank and solver checks are the meaningful replacements. The protocol deliberately fits the unclipped objective and clamps only at application, matching the existing affine controls.
2. Root has accepted, subject to final review, the advancement threshold **before any array inspection or solve**: at least +0.003 R² over both the source-matched affine and historical incumbent for either arm to merit a separate confirmation plan. The same +0.003 held-out versus in-sample spline contrast is required only for a transfer-specific advancement claim.

No other parameter search is proposed. There is no need to choose hidden width, activation, learning rate, epoch budget, optimizer seed, additional features or QC representation for this bounded comparison. Once final review passes, the next task may be separately authorized narrow implementation and preflight preparation, not immediate fitting or validation. The current document snapshot contains only protocol work and independent review.

## Evidence read for this proposal

Only published source text, decisions, receipts and aggregate summaries were inspected: Round03 DECISION/config/evaluation/affine-stage source, ROUND03_RESULTS.json, ANALYSIS_RECEIPT.json and independent scientific review; earlier round findings and the reviewed next-direction/history handoffs were already in task context. No label, prediction, index, checkpoint or cache array was opened, and no model/solver code was run. Completed campaign files were not modified.
