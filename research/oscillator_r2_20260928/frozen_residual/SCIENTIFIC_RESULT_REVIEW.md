# Independent frozen-head result and evidence-led priorities

Date: 2026-09-29. Decision: retain the frozen-head result as a small validation-only diagnostic. No promotion, test access or extension. Root has accepted preparation of one bounded convex Gram-feature probe; execution still requires implementation/preflight review.

## Completed result

| Checkpoint | Train R2 / SSE | Validation R2 / SSE |
|---|---:|---:|
| Initial eta0 | .646103561 /1069.507413 | .405294119 /94.205121 |
| Selected epoch2 | .647818205 /1064.325600 | .406789903 /93.968179 |
| Final epoch20 | .652200887 /1051.080735 | .404299111 /94.362736 |

Selected delta versus eta0 is+.001495785 R2, only.2515% less SSE. Molecule95% interval[+.0008635,+.0022007] and connectivity-group interval[+.0008293,+.0021849] are positive, but condition on validation selection over21 checkpoints and one seed. Positive bootstrap frequency1.0 does not replace the prespecified+.01 effect-size threshold. It is a small consistent selected-prediction difference, not the sought improvement.

The fixed equal-three ensemble remains ahead on this validation set: R2=.449421463, versus frozen single model.406789903; difference-.04263156, group95% interval[-.0614242,-.0253362]. Label that as a single-versus-ensemble comparison, with different inference cost, not architecture causality or fresh test evidence.

Full train SSE falls1.72% by epoch20 while validation peaks at2 and then worsens. The head can learn some corrections; it does not fail to optimize altogether. The small improvement does not establish convergence or exhaust nonlinear head capacity. There were zero clipped updates (largest recorded head norm.89086, threshold5), so clipping did not explain this frozen run's limitation. Validation E and base outputs are exactly invariant, as intended.

## What improved and what did not

At selected epoch2, true-f q90 SSE improves.515986, offset by.279045 worse SSE below q90. S10 supplies.078978 SSE improvement; S7 instead worsens.013293, and S8 improves only.002701. MAE worsens slightly from.01778561 to.01785150. About51.66% of molecules improve SSE, but the size of changes, not that count, determines pooled R2.

At final20, q90 SSE improves.81908 while its complement worsens.97669. S7 alone worsens.66271. About52.83% of molecules improve, yet total SSE rises.157615. Mean absolute correction grows from.001382 at selected2 to.001924 at20; the largest positive correction grows.01110 to.16139. The previously identified all-zero-target molecule14562 receives S7 corrections+.005513 at2 and+.161393 at20, adding.012163 and.560871 molecule SSE versus eta0. Freezing prevents the joint run's new base spike but does not guarantee a correction head extrapolates safely.

The prior joint run's changed base added20.01096 SSE, while its residual/fold reduced4.11390, for net+15.89706. Combined with this frozen result, preserving the base prevents most catastrophic continuation damage. It does not establish that existing h alone contains a large useful correction, or that one particular optimizer mechanism caused the damage.

## Decision branches resolved by the run

- Material frozen validation improvement would have justified confirmation while preserving the representation. That branch did not occur.
- Falling full-train error with early validation peak supports limited useful correction followed by mild overfitting/extrapolation problems. This is the observed branch.
- Flat/nonfinite train loss or persistent inability to learn would have required an implementation/optimization check before an architecture claim. That branch did not occur; small train gains still leave optimization and function-class limitations unresolved.

## Ranked next options

1. **One convex frozen-feature comparison with tensor Gram information.** Compare h+native base_f against those same inputs plus the full within-state symmetric Gram of the32 original readout tensor channels. Both use one fixed train-only ridge solve and identical positivity projection. This retains the known-good base, supplies invariant information omitted by previous heads, and removes iterative optimizer/epoch-selection uncertainty for these linear residual models. The concrete protocol is frozen_gram_probe/FROZEN_GRAM_PROBE_PROTOCOL.md. Root accepted preparation; no solve/inference is authorized by this note alone.
2. **Stability-preserving full-model fine-tuning**, after the already-running lower-weight objective study and this information probe. Controlled slow/regularized movement may learn features that a frozen representation cannot. Existing joint continuation makes an unconstrained repeat unattractive; any future comparison must separate learning-rate/regularization effects from objective changes. No new fine-tuning configuration or grid is proposed here.
3. **Properly scaled native DetaNet versus MTO from scratch.** This eventually removes legacy pretraining-objective bias and tests routing/readout on the same DetaNet backbone. It remains scientifically valuable, but costs much more and does not isolate the immediate omitted-Gram-information question. The previous100epoch proposal remains deferred, not automatically launched.
4. **Coupled-state or allocation-aware heads.** These may help near-degenerate-state assignment in other molecules, but the dominant inspected failure has all ten true f values zero; permuting state allocation cannot reduce its SSE. Broadening into coupled heads before testing omitted invariant magnitude information has weaker current support.

The top recommendation is also the cheap frozen-feature regression baseline requested in the option review; it is not an additional parallel baseline sweep. No feature, ridge or state-specific hyperparameter search is included.

## Representation versus optimization

h is state-conditioned but does not contain every cross-channel tensor Gram entry used by original intensity. Previous direct/trace heads removed that information, and the residual MLP's correction also only saw h. Their failure is not evidence that scalar invariant readouts in general fail. Appending native base_f first creates an amplitude-aware control; appending Gram next tests whether finer invariant tensor information adds signal beyond that scalar magnitude.

The complete within-state32-channel Gram is sufficient to reconstruct the quadratic tensor magnitude used by the existing frozen readout together with its h-derived gates/beta and E. It is not a complete set of all molecular invariants or cross-state relationships. A linear ridge head may also fail to exploit nonlinear h-by-Gram interactions; a negative result therefore cannot prove an information ceiling. Feature count/capacity differs between the two arms and will be reported rather than concealed.

Legacy-objective bias remains: all frozen features came from eta0's original objective. A fixed-feature success identifies useful existing information; a fixed-feature failure cannot determine whether native-f training from scratch would create better features. Architecture/original also gave f4.508882 times the relative weight of the pending loss pilot's direct-f arm, so cross-study differences are objective-scale evidence, not architecture gains.

## Independent checks and provenance

Read completed histories, FIT, INDEPENDENT_RUN_AUDIT.json, FROZEN_RESIDUAL_RESULTS, FROZEN_VS_ENSEMBLE_VALIDATION, FROZEN_ERROR_DECOMPOSITION and WORSTCASE_14562_VALIDATION outputs. Independently recomputed initial/selected/final SSE and base+delta/abs identity from saved arrays, checked exact validation IDs/indices/raw truth and all-epoch clipping counts. Executor verified12 source hashes, both cache hashes, all20 orders, aliases, fixed base/E and stable GPU ECC. No training polling, new model inference, test data or sealed-source edits occurred in this review.