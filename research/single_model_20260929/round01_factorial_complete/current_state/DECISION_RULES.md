# Decisions after the first single-model factorial

Written before completion of the frozen pilot on 2026-09-30. This is a resource-allocation plan, not a claim of statistical significance. The original20epoch protocol and validation checkpoint selection stay unchanged.

## Required evidence

Complete all four arms or document a real failure. Review native and fixed-calibration pooled raw-f R²/SSE/MAE, all states, fixed train-derived bright tails, energy errors, full curves, epoch0 and last checkpoints. Confirm source/data/order hashes and resumability. Use every valid label. No test inference for pilot decisions. A bootstrap on checkpoint-selected reused validation is descriptive, not fresh confirmation.

Compare each arm's validation-selected best checkpoint as a recipe. Separately compare identical fixed epochs (including20) for a factorial interaction. Do not call different best epochs mechanistic synergy. Preserve all results, including no change and degradation.

## Promotion

- A control-schedule gain of at least0.003 native pooled validation R² over epoch0 is eligible for independent-seed confirmation.
- An architecture/penalty arm gaining at least0.003 over both its matched control and epoch0 is eligible for confirmation.
- Compare secondary predictions using the same historical fixed calibration constants for all arms, without refitting. Beating the native baseline alone does not establish superiority to the calibrated one-checkpoint recipe.
- Inspect bright-tail, state, energy, stability and compute tradeoffs before allocating confirmation runs. Do not hide regressions or exclude difficult states.
- Confirm using paired controls from independently trained seed23 and37 checkpoints, reporting each model separately. Starting source contracts must be verified. The final predictor is one model and one checkpoint; no prediction or checkpoint averaging.

## If the factorial does not improve validation

Do not extend the same runs automatically. Use the prespecified training-only representation audit and error-gap diagnostics to decide whether a new controlled question is justified.

| Completed-round evidence | Possible next controlled question | What it would establish |
|---|---|---|
| Training loss improves while validation raw-f deteriorates across all arms | Optimize only the new pre-CG adapter while freezing the original model, with an explicit fixed baseline and a suitably matched capacity control | Whether this adapter can improve readout without changing the learned backbone; distinct from prior post-readout h-only residual |
| Adapter barely changes while the entire original model moves | Adjust adapter-only optimizer scale under a frozen-base protocol | Whether limited adapter optimization rather than its functional form explains the null result |
| Raw-state decorrelation changes negligibly at the audited weak weight | A separately registered modest-strength ablation, scaled by training-only gradient evidence | Sensitivity to regularizer strength; does not justify imposing physical orthogonality |
| Raw-state decorrelation changes substantially but validation worsens | Drop or redesign that penalty rather than blindly increasing it | The tested feature decorrelation is unhelpful under these settings |
| Native-f error is poorly aligned with the original energy/trace objective | A gentle matched raw-f objective study with a new documented schedule/control | Different from the failed1e-4 loss continuation; no attribution to learning rate without controlled evidence |
| Error is concentrated in crowded states and existing methods remain weak | Consider a stable empirical shared response/state-mixing adapter only after synthetic safeguards | A model inductive bias, not identified quantum operators or exact NTO reconstruction |

The table does not authorize any new fit before completed results, independent review, a concrete next-round protocol, and the required archive/publication order. Positive correlations between energy gaps and errors do not prove causation or label ambiguity.

## Physical interpretation limits

Raw M is a learned representation, not an identified electronic wavefunction. Its orthogonality and phase conventions are hypotheses. Rotation-invariant signed0e gate inputs do not automatically ensure electronic state-phase covariance of the nonlinear adapter. The preflight phase check applies to the squared-cosine penalty. Current PSD output already includes coherent cross-channel interference. Missing MO/AO/X/Y labels preclude verified transition-density/NTO reconstruction. Direct eigendecomposition has demonstrated sorting/degeneracy/gradient failures; a stable factor mixer remains empirical and gauge-unidentified.

No verified fresh holdout exists in the current data. Any final historical-test result must be labeled reused-test evidence. Do not redefine independent seeds as fresh data confirmation.
