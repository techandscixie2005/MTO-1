# Retained-residual completion: scientific interpretation

Date: 2026-09-29. Scope: completed retained_residual only; no active-arm inspection, test access, training or sealed-file changes. Sources: completion_receipts/RETAINED_RESIDUAL_COMPLETION.{json,md}; completed arm history/config, selected and final CPU-loaded checkpoint state dictionaries; architecture train/model/objective source. Validation of 44 source hashes, prediction-label alignment and 21 GPU-health snapshots is inherited from the completed executor receipt.

## Conclusion

The selected model is unchanged eta0 at epoch 0, within numerical tolerance. This arm provides no selected single-model improvement. Its later trajectory is consistent with continued training fitting the training objective while worsening generalization, concentrated in S7/S8. It does not yet establish that the residual architecture is inferior: the matched original control is required to distinguish common objective/optimizer continuation sensitivity from an additional residual-path effect.

| Measure | Epoch 0 | Epoch 1 | Epoch 20 |
|---|---:|---:|---:|
| Validation native-f R2 | .405294130 | .397408782 | .304937854 |
| Validation f SSE | 94.205119 | 95.454207 | 110.102179 |
| Validation f MAE | .017785608 | .017456119 | .017538633 |
| Validation common objective | .589312536 | .595363039 | .684791883 |
| Online training common objective | unavailable | .337980487 | .208473240 |
| Validation E MAE | .091388266 | .088878269 | .092416393 |

Final SSE increases 16.8749%, despite MAE decreasing about 1.39%. Better average absolute error does not compensate for larger squared errors under the prespecified pooled-f objective. R2 declines immediately and generally through the run; late oscillations around .30-.31 supply no evidence of late recovery. Both native-f and common-objective selection retained epoch 0. The selected-minus-eta0 R2 difference of 1.15e-8 is numerical reproduction, not a gain.

Training logs store batch-size-weighted, online LE+Lf loss along a changing model trajectory. They do not store separate full-training f SSE or epoch-0 training evaluation. Their monotonic decline from epoch 1 to 20 supports improved training-objective fit, but cannot quantify a checkpoint-matched train/validation f gap or prove raw train-f SSE declines by the same amount. Avoid describing these combined values as train f R2 or train f loss.

## Where the validation degradation appears

Total added SSE is 15.897060. S7 adds 12.530601 (78.82%); S8 adds 2.675430; together they account for 95.65% of the net increase. S7 R2 changes .049940 to -.807068. S1/S3/S4/S6 retain small SSE improvements, so deterioration is not uniform across states. These are diagnostics, not a reason to remove states or alter the metric.

The fixed true-f q90 group has threshold .0546 and 6690 entries: SSE 60.525630 to 61.808167, only 8.07% of the net increase. The nested q99 group (threshold .2377; 671 entries) changes 31.612563 to 31.711455, only .62% of the increase. The complement below q90 changes 33.679489 to 48.294012, accounting for 91.93% of the increase. Bright groups overlap and must not be added together. This is not primarily worsening on bright TRUE targets; large false-positive predictions on weaker transitions are compatible with the summaries. Without final individual predictions, the number, identities, signed errors and connectivity concentration of the responsible cases are unknown.

At epoch 1, bright-q90 SSE improves by 3.788659 while its complement worsens by 5.037747, already enough to worsen pooled SSE. Thus the deterioration cannot be summarized as a simple failure to fit bright transitions.

E MAE improves initially, then ends just 1.13% above epoch 0; E SSE increases 2.80%. The 10% energy-review threshold is not reached. The main failure is f generalization, not gross energy collapse; aggregate E statistics cannot exclude local E/f coupling problems.

## Optimization and the residual path

Fresh Adam AMSGrad, fixed lr1e-4, no decay, clip5, full-model updates, 20 epochs / 37620 steps; final optimizer metadata confirms those settings. No scheduler lowered lr. Reset optimizer moments and the new variance-scaled native-f objective are shared interventions; neither their effect nor lr suitability can be inferred from this arm alone.

CPU checkpoint comparison confirms weights/state tensors changed in the core, MTO routing, CG, decoder trunk, energy/beta/tensor heads and residual head. This is not a head-only fit or a frozen tensor-base experiment. No exact parameter-count inference was made from state dictionaries because these also contain buffers.

Zero final residual-head weight AND bias gives delta=0 and exact initial f. At the first step, the final affine layer can learn while the preceding residual hidden layer has zero gradient through that head. The base path still receives the common objective gradient. Once final weights move, the residual hidden layer can train. Every initial signed prediction was strictly positive (minimum about1.88e-5), so abs has its positive derivative and its zero cusp does not explain initial failure.

At epoch 20, 2930/66860 signed outputs are negative (4.38%), and none are exactly zero. abs folds those signs into positive predictions; these are not negative emitted oscillator strengths or clipped dead outputs. Mean absolute delta_f is .005614; delta range is [-.086608,+.009878], compared with zero initially. There is no saved per-case attribution establishing that sign folding caused the SSE loss.

Signed-output maximum rises from 1.665143 to 3.604861. Because final delta_f is at most .009878, the modified base at the largest positive signed output must be at least 3.594982. Thus this extreme cannot be produced by the additive residual alone: the original trainable intensity path also changed substantially. Its precise error contribution still requires final per-case predictions. Do not equate a small residual coefficient with a preserved base function after training.

## One conditional next action

Wait for the matched original and other head summaries. If the completed comparison still leaves the cause of this decline ambiguous, propose one bounded validation-only attribution pass on frozen final checkpoints: evaluate residual base_f separately from abs(base_f+delta), alongside matched original, and report paired SSE changes by S7/S8 and fixed q90 complement. This would distinguish direct residual contribution from changes in the jointly trained base. It is a proposed diagnostic, not a training/configuration authorization. No immediate extension, learning-rate sweep or scratch baseline launch follows from this result.

If original shows comparable decline, common continuation/target weighting/generalization is implicated; if original remains stable and residual declines, the added pathway and its induced joint optimization become the stronger suspects. Neither pattern alone proves an intrinsic tensor expressivity ceiling. Existing direct/trace arms may further clarify this, subject to their information-loss and initialization caveats.

## Verification limits

Selected epoch-0 predictions were independently checked by the executor. Final epoch-20 individual predictions were not separately saved; final metrics are cross-checked against history, status and committed last-checkpoint state, not independently regenerated here. This audit loaded model/optimizer checkpoint metadata on CPU only and performed no inference. All conclusions remain exploratory on a previously used validation split. No test labels or active run outputs were opened.
### Replay information value and limits
One final-epoch validation replay is worthwhile only after the matched summaries, when a healthy GPU is free and no active job is affected. Within that single pass, preserve per-entry raw truth, frozen epoch-0 prediction alignment, base_f, delta_f, signed and emitted f, molecule/state IDs. Report SSE-change concentration in the largest-error cases and the fixed cross-table S7/S8 versus other states, below-q90 versus q90, and negative versus nonnegative signed outputs. This can distinguish rare weak-target overprediction from broad bias and directly quantify the residual's inference contribution relative to its jointly changed base. Use the frozen epoch-0 arrays already available; no refitting or threshold search is needed.

Current aggregate histories establish S7/S8 concentration and below-q90 concentration SEPARATELY; they do not establish their joint intersection, error-sign distribution, or number of exceptional cases. The proposed replay supplies that missing evidence and independently regenerates final metrics.

For nonnegative truth, abs(signed) cannot worsen squared error relative to leaving a negative signed prediction negative: the reduction is exactly4*truth*abs(signed) on negative entries. Thus negative signed counts alone do not diagnose a harmful output activation. Harmful excursions must be assessed relative to the base or frozen predictor. The largest final positive output also cannot come from folding the most negative signed value, whose magnitude is only .046151. This narrows the mechanism hypothesis before any replay.

A concentrated error increase in the changed base would favor investigating joint continuation/generalization over blaming the small additive correction; a broad harmful direct residual contribution would make the residual path more relevant. These are conditional interpretation rules, not a chosen optimizer/head intervention. One replay cannot establish the counterfactual effects of retraining without a component; matched control evidence remains necessary. No replay was run or authorized by this note.