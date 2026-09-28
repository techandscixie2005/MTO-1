# Frozen three-model oscillator-strength ensemble: exploratory historical-test result

## Objective and fixed setup

Hypothesis: the three completed MTO η models have complementary native oscillator-strength errors, so an equal-weight average can improve pooled raw-f R². The models are `mto_eta0`, `mto_eta01`, and `mto_eta1` from `experiments/qm9s_eta_Ef_20260926`. Each contributes its **saved native f prediction**, computed from its own predicted energy and tensor. The ensemble averages those three f arrays with weights exactly 1/3 each. No model weights, checkpoint choices, or data splits changed. Three model forward passes are needed in deployment.

The validation diagnostic `ensemble_metrics.json` aligned 6,686 molecule IDs, source indices, raw f labels, masks, and state-slot order exactly. On the validation split, η=0 alone had pooled f R² **0.405294** (SSE 94.205121) and the equal three-model average had **0.449421** (SSE 87.215074), a +0.044127 R² difference and 7.42% SSE reduction. Fixed equal weights beat the tested fitted-simplex cross-fit score (0.443012 R²). The equal three-model rule, source/checkpoint provenance, validation result, code hashes, and bright thresholds were written to `FROZEN_CHOICE.json` before this script opened any test prediction file. Its SHA256 is `ed917f216263d8e949192f83e5e9746551aebf17f79ed704252b87f92dec0dd3`. An independent validation review is recorded in `ENSEMBLE_REVIEW.json`.

## One exploratory historical-test comparison

Both methods were scored on the same 6,686 test molecules × 10 state slots (66,860 original printed oscillator strengths), with no clipping. All three test prediction files matched exactly on IDs, source indices, raw truth, and masks. Validation/test IDs and source indices were disjoint. Pooled R² uses one mean and one SSE/SST across all valid molecule-state values.

| Metric | MTO η=0 baseline | Equal three-model ensemble | Change |
| --- | ---: | ---: | ---: |
| Raw native-f R² | 0.455815 | **0.485102** | **+0.029287** |
| Raw-f SSE | 103.193194 | **97.639597** | 5.38% lower |
| Raw-f RMSE | 0.039286 | **0.038215** | 2.73% lower |
| Raw-f MAE | 0.017716 | **0.017265** | 2.54% lower |

A paired bootstrap resampled molecules, retaining all ten state slots (2,000 samples, seed 20260928). Its percentile 95% interval for ΔR² is **[+0.004705, +0.055965]**; 99% of resamples had a positive difference. This describes sampling uncertainty on this reused split. It does not cover retraining variation or validate a new unseen test set.

The ensemble improves test per-state R² in states 1–6, 9, and 10. It worsens state 7 from 0.671835 to 0.651923 and state 8 from 0.351520 to 0.338801. For bright transitions defined by the **validation** truth 90th percentile f≥0.0546, test SSE falls from 69.254332 to 68.889938 across 6,653 entries. At the validation 99th percentile f≥0.2377, it rises from 41.176598 to 42.655354 across 672 entries. Full per-state metrics and source hashes are in `EXPLORATORY_TEST.json` (SHA256 `c85de6aae489ea43cf8b0559b29813d86e670878bcc6fe1c67cc4fac05b9e6a6`).

## Interpretation and next action

The fixed ensemble clears the project's exploratory +0.02 R² improvement target on this historical test and improves pooled RMSE/MAE. It still requires confirmation: the test split had already been examined in previous campaigns, the three source checkpoints were selected on this validation split, and the upper 1% of bright-transition errors worsened. Preserve these exact weights and predictions for reproducibility. Next, compare against the queued matched objective screen on validation, then replicate promising methods across matched seeds and a newly sealed molecule-disjoint holdout before claiming a robust generalization gain. Do not tune this ensemble using the historical test.


## Connectivity-group bootstrap sensitivity

The original split grouped molecules by a conservative RDKit connectivity key. The 6,686 test molecules contain 6,671 such groups; 14 groups contain more than one test molecule (largest group: three). Resampling whole connectivity groups rather than individual molecules, while retaining all ten state slots per molecule, gives a paired ΔR² 95% interval of **[+0.003586, +0.056691]** (2,000 replicates, seed 20260928); 98.75% of resamples are positive. The sign of the interval is unchanged. This sensitivity is stored in `GROUP_BOOTSTRAP_SENSITIVITY.json`; it does not alter the frozen ensemble or the original molecule-bootstrap result.

