# Round02 closeout and next preparation

## Decision

Do not promote either frozen-adapter recipe. Do not extend the same runs or raise their learning rate without a new controlled rationale. The strongest verified eligible predictor remains the previously packaged calibrated eta0 checkpoint.

| Frozen-F objective | Selected epoch | Native validation R² | Gain over own epoch0 | Same fixed calibration R² |
|---|---:|---:|---:|---:|
| Original LE+Ls | 2 | 0.4054078302 | 0.000113700 | 0.417538722 |
| LE+Lf | 1 | 0.4053875246 | 0.000093404 | 0.417865155 |

Both gains are below the prespecified 0.003 allocation threshold. The raw-f objective is 0.000020306 below its matched trace-objective control. Both calibrated scores are below the fixed calibrated anchor, approximately 0.41811924. Descriptive validation bootstrap intervals for the tiny native gains include zero; reused validation is not fresh confirmation.

Both runs completed the fixed20 epochs and37,620 AMSGrad steps normally. All64 frozen source hashes, all20 matched data-order hashes, earliest native-SSE selection, original parameter/buffer freeze, selected checkpoint arrays, standalone geometry-only loader, and resumable checkpoints passed review. The trained predictor uses one self-contained checkpoint. No test inference was performed.

At epoch20, native R² is 0.403125542 / 0.402157980. The adapter is active: mean audited input changes are approximately6.05% /4.97%. No epoch clips gradients; maximum norms0.516 /2.280 stay below5. Energy and true-bright-tail errors improve modestly while native MAE worsens. False-bright-bin SSE increases1.156 /1.343, with S7/S9 contributing regressions. These findings do not support describing the null result as an inert adapter or inadequate optimization solely because the learning rate is small.

## Training-error audit

A single bounded cached forward examined all1,203,550 TRAIN labels, including22,646 valid zeros, without updates or validation scoring. Native TRAIN R² is0.6461035600 and MSE0.000888627323. Descriptive global affine coefficients are slope1.0035907458 and intercept0.0000905206. Several higher states favor amplification on TRAIN. These differ from the already-fixed historical validation coefficients0.8511830211 /0.0037251854.

The coefficients are statistical diagnostics, not deployed models. This supports a difference between the source model's in-sample error distribution and its error distribution on unseen molecules. It does not establish that this alone caused either frozen-readout failure. Historical frozen h-only raw-f training is a relevant prior result, not an untried method.

## Next authorized work: feasibility and preparation only

Investigate a controlled readout-transfer study using a temporary source model trained from label-clean initialization on80% of the original TRAIN molecules. Reserve the other20% solely to generate out-of-sample source predictions for readout training. The outer train/validation/test assignments remain frozen. Fine-tuning the current full-TRAIN checkpoint on a subset would not produce label-clean held-out predictions and is prohibited for this purpose.

Tentative source budget is the original eta0 architecture and LE+Ls recipe for a fixed33 epochs. Source normalization must use only its fitting subset. Calibration-subset, outer-validation and test labels must not select the temporary source checkpoint. Independent source review must verify initialization, label access, statistics, split identities, resume behavior and computational cost before a final protocol is frozen.

The proposed matched readout comparison uses the same reserved20% labels with either held-out-source predictions or the existing full-baseline's in-sample predictions. Both eventual predictors would use the same full-baseline network and one readout in one checkpoint. No source-model prediction averaging or source-model dependency at deployment is allowed. Inputs must have transferable observable meaning; arbitrary hidden coordinates from independently trained networks are unsuitable without an audited alignment method.

The smallest next stage uses only two-parameter global affine moment fits on the same reserved20% labels: one from held-out-source predictions and one from full-baseline predictions. Both apply the predeclared rule max(0,a*f+b) to the SAME full baseline at evaluation and deployment. The existing historical validation-fitted affine map is an exposed comparator and is already optimal or nearly optimal within this simple family on that validation set; this stage tests transfer, not a promised new best score.

Proceed to preparation of a nonlinear observable readout only if the held-out-trained map gains at least0.003 native validation R² over BOTH the in-sample-trained map and the unmodified native anchor, after examining per-state and bright-tail tradeoffs. Merely approaching the existing calibrated reference does not justify an independent-seed promotion or a new best-model claim. No nonlinear readout fit is approved by this decision.

Use internal split seed20260930 with the original molecule-grouping policy, derived from original TRAIN IDs without labels. The independently verified metadata-only dry run keeps all266 unresolved groups (278 records) in source fitting, shuffles sorted eligible group keys, and takes a whole-group prefix until at least24071 held-out rows are assigned. It yields exactly96284 fitting and24071 held-out molecules; no group is broken and no valid molecule is excluded. Preserve original TRAIN order within each part. Identity-audit SHA is9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4; fitting-index SHA b7b00dfe514ae57f3dd609a60dc259625007a5f43288ff75e2c83a3a7e71da37; held-out-index SHA9547ca89feff3301f5e4c6fcd736be7a9186878a9fcb1fec8695c5c12d60707b. New split generation must reproduce those identities before launch.

The temporary source uses seed11, original lr0.001/AMSGrad/batch64/clip5/FP32/weight-decay0 and fixed33epochs. The original patience50 scheduler cannot reduce the learning rate within this budget, so fixed lr0.001 is equivalent for that prefix. No early stopping or held-out-label selection. Preserve atomic resumable checkpoints and the fixed epoch33 source.

This decision authorizes implementation/preflight preparation, not a new production fit. A separate exact protocol, independent review, archive-first preparation publication and resource admission are required before launch. Differences in source training-set size and source quality are real confounders that the study must report. A single internal holdout is not full cross-fitting and is not fresh outer holdout confirmation.

## Other hypotheses checked

- No matching Adam state is retained for eta0 epoch33. The saved last optimizer belongs to the completed epoch236 trajectory. Do not combine it with epoch33 weights or claim a matched preserved-moment experiment. No reconstruction run is authorized.
- A neutral PSD covariance over point charges passes geometric symmetry tests but cannot represent dipoles outside the nuclear-coordinate span. Reject it as a general dipole-tensor replacement. It is at most a partial empirical prior; no fit is authorized.
- A canonical-root state-factor mixer remains a bounded mathematical feasibility question, secondary to the current evidence. It is not an identified electronic response operator and is not approved for fitting.
- Available QC labels still do not support verified full-TDDFT transition-density/NTO reconstruction. No new quantum-chemical inputs may be required at inference.

## Required closeout order

Complete the independent review of the TRAIN moment supplement and freeze the concise completed report. FIRST download all lightweight round02 records to D:\MTO\archives\ and verify them. THEN inspect an explicit byte-exact staged allowlist, commit and push as a descendant of a4c2fabd837bea5ae7eced79d11de797af3900bf. Include all arms, objectives/settings/source/logs/results/analyses/reviews, checkpoint metadata, this decision and bounded rejected hypotheses. Keep checkpoints, optimizer states, raw labels, caches, prediction arrays and credentials off the archive/publication path.

The overall accuracy objective remains unfinished. No independent-seed confirmation or historical-test evaluation is justified by this round.
