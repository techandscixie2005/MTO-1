# Independent architecture decision and next bounded experiment

Date: 2026-09-29. Decision: promote NONE from the completed four-arm architecture screen. Retain eta0 as the single-model reference. No historical test evaluation is justified by this round. Root has authorized preparation of one frozen-eta0 residual-head-only experiment, subject to implementation review/preflight; this note launches nothing.

## Matched results

| Arm | Initial R2 | Selected epoch / R2 | Final20 R2 | Connectivity-bootstrap95% interval for selected delta vs original |
|---|---:|---:|---:|---|
| original | .405294 | 0 / .405294 | .311987 | zero by construction |
| retained_residual | .405294 | 0 / .405294 | .304938 | numerical zero |
| direct_f | .334568 | 1 / .369250 | .305924 | [-.10728,+.00558] |
| independent_trace | .335515 | 1 / .372023 | .314608 | [-.09716,+.00670] |

Direct and trace selected point estimates are worse by .036044 and .033272 R2 (6.06% and5.59% extra SSE). Their intervals cross zero; this is no evidence of a useful positive gain and also not a precise proof of universal architectural inferiority. Molecule intervals are similar: [-.10968,+.00756] and[-.09941,+.00824]. Connectivity grouping changes little because6686 molecules form6671 groups, only15 repeated groups of size2. Both intervals condition on validation-selected checkpoints, omit training-seed variability, and do not correct selection among epochs/arms.

Residual-minus-original selected delta1.46e-8 and bootstrap positive fraction.969 reflect floating-point reproduction of the same epoch0 predictor. The effect-size criterion is essential: positive bootstrap frequency alone would produce a false promotion. None meets the prespecified+.01 R2 threshold. Direct/trace best epoch1 and late deterioration also fail the extension rule; no extra epochs are warranted.

## Interpretation

Original continuation already loses .093307 R2 by epoch20; residual loses .100356, only .007049 lower than original at the final checkpoint. Most deterioration is therefore shared continuation sensitivity under this objective/optimizer, not an effect unique to the added residual. Both train online combined objectives decline (.33806 to.21119 original; .33798 to.20847 residual), while validation objectives worsen (.58931 to.67807/.68479). This supports deteriorating generalization during continued fitting. Separate checkpoint-matched train f SSE is not stored, so do not assert an exact train-f generalization gap.

For original, S7/S8 account for14.26325 of14.78047 added SSE; below-q90 targets account for14.13024. Residual corresponding increments are15.20603 of15.89706 and14.61452 below-q90. These are separate aggregate partitions; their intersection, exceptional-case count and signed errors require the reviewed final replay. At selected direct/trace checkpoints, S7 alone contributes about4.98514/4.53556 of5.70960/5.27042 added SSE. Neither pooled failure nor its mechanism should be summarized simply as failure on bright TRUE transitions.

Selected direct/trace E MAE improves about1.67%/1.45%; final E MAE changes across arms are only about+1.1% to+2.0%. The common10% energy flag is not reached. Modest energy improvements do not overcome worse f R2, the primary objective. Raw/A-derived truth sensitivity is negligible at the scale of these losses.

This round uses train Var(f)=.002510981243894732, giving4.508882 times the f coefficient of the prior loss pilot's direct-f denominator relative to unchanged LE. Original here is therefore essential. The pending lower-weight loss comparison must be interpreted as objective-scale evidence, not as an architecture advantage; it is not silently replaced by this completed round.

Direct/trace initially approximate the teacher rather than reproduce it: train-only warmup normalized teacher errors.06344/.06446 remain, and validation starts about.070 below eta0. They also remove tensor Gram information not fully represented in hidden h. Their near-matched trainable counts (1,551,996 versus original1,552,092) do not remove these information and initialization confounds. Residual adds4161 parameters (total1,556,253), preserves the full initial function, but then permits all base weights to change. None of these results establishes a tensor expressivity ceiling or a need for a new backbone. MTO already shares the DetaNet backbone; the previously proposed100epoch scratch readout pair is deferred.

## One next training experiment

Proceed, after the completed-round integrity audit and one reviewed final20 residual replay, to the single frozen-eta0 residual-head-only experiment specified in frozen_residual/FROZEN_RESIDUAL_PROTOCOL.md. Copy the exact existing residual initial head, freeze h/E/base, preserve training orders,20epochs, Adam AMSGrad1e-4 and the same raw-f variance loss. The frozen epoch0 model is the zero-update control; the completed jointly trained residual is the matched-seed reference for the intentional freezing intervention. No new jointly trained rerun is needed.

This tests whether existing eta0 features contain a useful correction that full-model updates failed to preserve. It is cheaper than a new scratch architecture pair and directly follows the observed shared base drift. A material gain would prioritize frozen correction for later validation confirmation; failure would bound this specific head/feature/optimization setup, not prove no useful correction exists. No hyperparameter grid, extra epochs or test evaluation belongs to this experiment. The pending loss pilot remains relevant context before broader commitments, but root has authorized this small diagnostic's preparation.

The final residual replay remains useful despite the shared-control failure: it will expose whether a few weak-target predictions dominate the decline and separate inference-level base/residual contributions. It must retain its noncausal interpretation and cannot promote final20. Model arrays/checkpoints stay on the server.

## Evidence and limits

Inspected completed architecture/ARCHITECTURE_RESULTS.{json,md}, all four completed histories, architecture protocol/config and analysis_addendum/ARCHITECTURE_VALIDATION_ADDENDUM.{json,md}. The executor separately audits source/checkpoint/label/resource integrity. No active loss-job polling, test access, model inference, training or sealed-file edits occurred in this scientific review. Detailed residual and replay reviews remain in this completion_receipts directory.