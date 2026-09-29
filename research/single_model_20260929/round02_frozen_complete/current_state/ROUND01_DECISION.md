# Round01 decision and round02 preparation

## Completed result

The four-arm, 20-epoch full-model pilot completed normally. Control, right-side shared equivariant nonlinear F, weak normalized raw-state decorrelation, and their combination all select epoch0. Selected native tensors are the starting eta0 checkpoint. There is no verified pooled raw-f accuracy gain and no candidate to promote to independent-seed confirmation.

Both proposed interventions acted during training: F changed its input by approximately 0.75–0.78% on the fixed training audit, and decorrelation reduced raw-state overlap. Their absence of a gain cannot be explained by claiming they were inert.

For the control at epoch20, pooled validation SSE increases by 6.1206. The true-dim/predicted-bright bin at the fixed TRAIN q99 threshold f=0.2412 increases SSE by 10.0074; the other three brightness bins improve. S7 and S8 contribute increases of 9.3017 and 1.5589, while the other eight states improve. The largest 0.1% of errors account for 22.83% of baseline SSE and 31.95% at epoch20. All four arms show this pattern. These are post hoc diagnostics on reused validation, not independent evidence or a causal explanation.

Energy error, MAE and true bright-tail RMSE can improve while these false-bright errors worsen pooled R². Global S7/S8 summed-error SSE also worsens, so simple anticorrelated redistribution does not explain the full regression. Tight-gap cancellation remains a local hypothesis. The existing gap-bin audit uses selected epoch0 predictions, and differences between bins also reflect state composition and brightness; it cannot establish the cause of epoch20 deterioration.

## Next question

Can the same F improve readout accuracy when all original representation-producing and decoder parameters remain frozen, and does direct raw-f training help this restricted correction?

Freezing limits which parameters can change, but does not guarantee control of rare outliers. This study differs from the completed full-model continuation in its trainable parameter set. Historical frozen h-only residual work already used raw-f MSE, so frozen training or raw-f supervision alone is not a new idea. The new location is before CG, with coupled E and A effects through the frozen decoder.

## Authorized preparation: round02

- Two active arms: frozen-base F with original LE+Ls, and the same frozen-base F with LE+Lf.
- Lf is all-valid-state MSE against raw printed f divided by the fixed TRAIN population variance, reported historically as 0.002510981243894732; verify its full-precision provenance before source freeze. Coefficient is 1. No fitted gradient-rescaling coefficient.
- Existing 4016-parameter shared right-side F, identity initialized. Original model parameters and all buffers are frozen. M0 and original raw bypasses retain the original architecture. E and A remain coupled through the frozen decoder in both arms.
- Raw-M decorrelation is constant under this freeze and is omitted. Do not add clipping, robust loss, selective exclusions, output caps, a new E branch, or label-driven sampling.
- Same Adam AMSGrad, learning rate 1e-5, batch64, seed/order11, weight decay0, clip5, FP32, 20 epochs. Epoch0 is eligible. Save atomic resumable best and last checkpoints.
- Frozen raw-M caching is a training optimization only. Verify real TRAIN examples, full-versus-cached E/A/f, F gradients, and the next Adam update. Verify original parameters and persistent/nonpersistent buffers remain unchanged. Record cache identity/coverage and keep all cached arrays on the server.
- Select by native pooled validation raw-f SSE across all valid labels. Apply the same historical calibration constants secondarily without refitting. Report pooled, per-state, true-bright-tail and false-bright errors, energy errors and training behavior. No test scoring.
- A gain of at least 0.003 native R² over epoch0 makes either frozen-F recipe eligible for independent-seed confirmation. An objective-specific promotion requires at least 0.003 over its matched frozen-F objective control as well; do not attribute a common gain to raw-f supervision. Choose one recipe using validation and reported tradeoffs before confirmation. Beating the native anchor alone does not establish superiority to the calibrated baseline. This comparison does not identify an advantage of nonlinearity without a linear adapter control.
- The unchanged O(3)-equivariant adapter is not established to obey electronic sign covariance. Its interpretation remains empirical. No claim about NTOs or exact transition-density reconstruction is supported by currently available labels.

## Execution and publication gates

This decision authorizes implementation and preflight only. Complete independent review, exact source/config freeze, one-checkpoint geometry-only inference checks, and an archive-first preparation publication before launch. Monitor assigns healthy free GPU resources and registers distinct owned jobs. Preserve unrelated jobs and existing checkpoints.

First finalize round01: lightweight server records must be downloaded and verified under D:\MTO\archives\ before exact allowlisted staging, commit and push. The next commit must descend the remotely verified research branch (currently 0e612523cd99594b6ce05b491b8de1b18d8a2e66). Include this decision, all results including null arms, independent review, diagnostics and current limitations. Never publish weights, optimizer states, arrays, caches or credentials.

## Continuing reference and limitations

The existing calibrated eta0 single checkpoint remains the strongest recorded eligible recipe: validation-refitted R² 0.4181192453, historical OOF 0.41665766, reused historical-test 0.459667233. Native eta0 validation R² is 0.405294118. These figures have different selection/exposure histories and are not fresh confirmation. No verified fresh holdout is available. The research objective remains unfinished.
