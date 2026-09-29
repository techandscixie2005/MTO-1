# Round03 decision: transfer supports preparation, incumbent retained

Written after the fixed comparison completed on 2026-09-30 Asia/Shanghai. Independent terminal integrity and scientific reviews passed. Include both reviews before publication. This decision does not authorize another fit.

## Result and retained recipe

| Fixed predictor | Validation pooled raw-f R² |
|---|---:|
| Full eta0, native | 0.405294124 |
| Affine fitted to full-baseline in-sample predictions | 0.402024351 |
| Affine fitted to held-out-source predictions | 0.416374515 |
| Existing validation-fitted affine | 0.418119244 |

The held-out-source map exceeds the in-sample map by 0.014350164 and the native anchor by 0.011080390. Both exceed the prespecified 0.003 preparation threshold. It trails the incumbent by 0.001744729. Retain the existing calibrated eta0 as the strongest recorded eligible recipe. There is no new-best claim or independent-seed promotion from this round.

The transferred map is `max(0, 0.9025853252242239*f + 0.0018285582112617521)`, applied directly to the original full baseline's native prediction. It replaces, rather than composes with, the historical affine. Each export contains that same full model and one coefficient pair; no temporary source or prediction averaging is used in deployment.

Tradeoffs remain in the decision. Relative to native, held-out calibration improves eight states and slightly worsens states 1 and 5. Its MAE rises from 0.017785608 to 0.017805840. TRAIN-q99 tail RMSE rises from 0.219912917 to 0.227281574. Relative to the historical calibrated incumbent, it has lower MAE and tail RMSE but lower pooled R². No difficult states, zeros or tail labels were excluded.

## Interpretation limits

The clean source used 96,284 original-TRAIN molecules for exactly 33 epochs and 49,665 updates, with fit-only initialization statistics and no validation selection. Both affine fits used the same remaining 24,071 molecules and all 240,710 valid printed-f labels. Their coefficients were frozen before one outer-validation comparison over all 66,860 labels.

On the shared internal calibration molecules, the clean source's native R² was 0.429408124, versus 0.661066543 for the full baseline that had trained on those molecules. The experiment supports this one transfer procedure. Training size, source quality and optimization history differ, so the result does not isolate training membership as a cause or explain previous adapter failures. It is one internal holdout, not full cross-fitting. Outer validation is historically reused; no new historical-test inference or fresh holdout confirmation occurred.

Source execution and validation placement on GPU1 were verified. A short-lived MTO process was observed on GPU0 during the affine-fit interval, without enough captured identity to prove the fit's complete device placement. Preserve that resource-contract uncertainty. The fit was not repeated. Validation was independently reviewed, bound to GPU1's UUID before Python startup, and observed on GPU1. No unrelated process was signaled or modified; performance impact was not measured.

## Next decision: smallest identifiable preparation

The passed allocation gate supports a proposal for two matched small shared **f-only** nonlinear readouts: one learns from the clean source's held-out predictions, the other from the full baseline's in-sample predictions on the same molecules. Both would deploy with the unchanged full baseline in a single checkpoint. The existing frozen affine maps supply same-input linear controls.

Prepare a concrete protocol before any fit: exact function and parameter count, common feature scaling, initialization with live gradients, optimizer/budget/order/seed, validation-only checkpoint rule, unchanged raw-f MSE and nonnegativity convention, and complete per-state/tail reporting. Preflight must establish frozen base tensors, finite gradients, identical paired data, and self-contained geometry-only inference without the temporary source or caches. Do not use hidden M/h coordinates from independently trained sources without alignment.

Do not add predicted energy or context features to a comparison described as isolating nonlinearity. An E/f readout needs a same-input linear E/f control. The documented historical calibration family tested scale, per-state shrinkage and affine maps, not a flexible f-only map; the failed hidden-h residual is a different, retained negative result.

This heartbeat completes Round03 and its archival work. It launches no new fit and performs no nonlinear-head implementation. The next preparation remains a subsequent research step under the persisted decision. A new fit requires its own frozen protocol, meaningful preflight, independent review, archive/publication and explicit execution decision. No budget extension, test scoring, checkpoint averaging or automatic seed launch follows this result. Scratch F/decorrelation and AO-integral architecture work remain deferred.

## Closeout order

Finish the independent scientific review and reproducible report, then FIRST download and hash-verify all lightweight source/settings/logs/results/analysis/decision records to `D:/MTO/archives/single_model_20260929/round03_transfer_complete`. Only then inspect exact staged files, commit, non-force push and verify remote ancestry from `1b34e839` or its verified descendant. Include the resource observation and supplemental AO/next-direction reviews. Checkpoints, optimizer states, raw data, index arrays, prediction arrays and caches remain server-only.
