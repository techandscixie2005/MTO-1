# Interpretation of the completed four-arm pilot

## Decision supported by this round

None of the four continuation recipes meets the prespecified confirmation threshold. Every native-raw-f-selected checkpoint is epoch 0. The few parts in 1e8 between selected outputs are GPU floating-point replay differences, not model improvements. Keep the original eta0 checkpoint as the native reference and its fixed historical calibration as the stronger eligible deployment reference; do not promote an epoch-20 model.

| Arm | Selected native R2 (approximately) | Fixed epoch20 native R2 | Fixed epoch20 calibrated R2 |
|---|---:|---:|---:|
| Control | .405294116 | .366655307 | .394554483 |
| Right adapter | .405294117 | .366210889 | .394237328 |
| Raw-state decorrelation | .405294117 | .366738959 | .394629619 |
| Both | .405294115 | .366296425 | .394313638 |

The selected fixed-calibration benchmark remains approximately .41811924. Its coefficients were already fitted using historical validation; this is not fresh holdout confirmation. All new results use the reused frozen validation set and no test inference. The exact scores, checkpoint hashes, state/tail metrics, bootstrap caveats, and aligned factorial contrasts are in ROUND_RESULTS.json.

## What changed during continuation

For the control, normalized validation LE+Ls improves from .172227 to .170812 in the first epoch while native raw-f R2 decreases from .405294 to .403075. This demonstrates that improving the existing validation objective need not improve the requested metric. By epoch20, training trace loss falls from .079956 (epoch1) to .063166, while validation trace loss rises from .144046 (epoch0) to .152694. Validation energy loss improves from .028181 to .025668. These patterns are consistent with both objective mismatch and later trace generalization loss; they do not identify a single cause.

Pooled SSE deterioration is concentrated rather than uniform. Control MAE improves (.017786 to .017452), and true q90/q99 bright-tail RMSE improves (.095359 to .092605; .219913 to .212540). State7 SSE grows by9.30165 and state8 by1.55888 while every other state's SSE falls. The net pooled SSE increase is6.12064.

The post-hoc FALSE_BRIGHT_RESULTS.json partitions every valid label into all four combinations of true/predicted f above or below the fixed TRAIN q99 threshold .2412. Control true-dim/predicted-bright SSE grows12.60368 to22.61109, accounting for +10.00742 SSE; the other three bins all improve. The count in that bin changes only112 to113. Maximum predicted f grows1.66514 to3.34745, and the largest67 errors (0.1% of labels) carry22.83% to31.95% of total SSE. All four arms show the same pattern. No labels were removed, downweighted, or changed for evaluation. This evidence supports investigating extrapolation of large predictions; it does not justify clipping at a threshold chosen from validation.

At the fixed256-molecule TRAIN diagnostic subset, mean raw-state overlap squared changes from .438463 initially to .421382 for the control, .421355 for the adapter, .412794 for decorrelation, and .412703 for both. Thus the penalty has a detectable feature effect. Mean adapter change relative to raw Mk is .007543 for the adapter and .007793 for both (99th percentiles .012610 and .013038). The adapter is active, not gradient-dead. Native-f output drift RMSE is about .00940 in all four arms; energy output drift RMSE is about .0424eV. These are agreements with initial predictions, not predictive accuracies against labels. They do not identify feature overlap with physical wavefunction orthogonality.

## State crowding is an unresolved hypothesis

The baseline gap-stratified audit retains all66860 validation labels and uses TRAIN-only gap quantiles. Smaller true gaps co-occur with greater error, but state and intensity distributions also differ between bins. Do not interpret this association as demonstrated label ambiguity or state mixing.

The post-hoc state-error audit gives a sharper limit: for S7/S8 across all6686 molecules, centered error correlation changes from -.1956 to +.0385 and summed-error SSE worsens23.6963 to41.8414. Global regression therefore is not explained solely by cancelling redistribution between those states. For the519 S7/S8 pairs with direct true gap<.0287eV, summed-error SSE is1.3033 initially and1.3543 at epoch20, substantially below their summed individual SSE (ratios .460 and .415). Local anticorrelated errors remain scientifically interesting, but are not sufficient evidence for relabeling, a physical mixing operator, or replacement of the primary metric.

## Smallest useful next study (proposal; not authorization)

Use the same identity-initialized pre-CG F with every original parameter and every buffer frozen, retaining raw M0 and both original skip paths. Train only its4016 parameters. This tests a distinct restriction after the common full-model continuation drift; the current null result does not prove this will work. Raw-M decorrelation becomes constant under this freeze and should not be included as an active factor.

The smallest informative comparison is two matched frozen-F arms: (A) original LE+Ls; (B) LE plus printed raw-f mean squared error divided by the historical TRAIN population variance .002510981243894732, with coefficient1 and exact variance provenance verified before freeze. Retain AMSGrad1e-5, batch64, order11 and20epochs for the first bounded comparison, keeping the current schedule so freezing is the main change in arm A. Use identical F initialization and all optimizer settings in the two arms; the unchanged identity model is an epoch0 reference, not a trained capacity control. Root chose fixed coefficient1 for historical comparability; a TRAIN-only gradient audit reports magnitudes without rescaling the objective. Do not sweep learning rates or that coefficient on validation. If causal attribution to nonlinearity is later needed, add a simpler active right-channel linear adapter; its lower parameter count must be disclosed.

Historical precedent must be stated correctly: the previous frozen h-only residual already optimized raw printed-f MSE with TRAIN variance normalization, used4161 parameters and AMSGrad1e-4, and obtained only about+.00150 R2. A frozen pre-CG raw-f study is new because it changes the input to the bilinear interaction and can affect both E and A through the frozen decoder. It is not the first freezing or raw-f-loss experiment. Do not call F strictly dipole-only.

Caching all raw M for TRAIN is technically feasible because the original network is frozen, dropout is zero, and there is no geometry augmentation or batch-statistic normalization. The existing synthetic CPU feasibility PASS is insufficient for launch: require actual TRAIN/GPU full-versus-cached E/A/f, gradient and next-update parity; ID/row/hash coverage; unchanged parameters and nonpersistent buffers; one shared post-M function; and a full geometry validation epoch0 replay. Cache data stays server-only. Export the complete base-plus-F model so inference still needs geometry alone.

Any next study needs a separate preregistration, immutable sources, independent review, archive-first publication, and fresh healthy-GPU admission. Keep native pooled raw-f validation selection with epoch0 eligible, fixed calibration secondary, all-state and bright/false-bright reports, and no test inference. A gain must pass independent starting-seed confirmation before a general accuracy claim. No next fit is authorized by this document.

## Records

Primary evidence: ROUND_RESULTS.json, ANALYSIS_RECEIPT.json, runs/*/history.jsonl, MECHANISM_RESULTS.json, GAP_AUDIT_RESULTS.json, FALSE_BRIGHT_RESULTS.json and FALSE_BRIGHT_RECEIPT.json. The false-bright/error-coupling analysis was requested after the round completed and is explicitly post hoc. It replays each fixed epoch20 checkpoint; no prediction arrays or molecule IDs are exported. Its bins partition all valid labels and the summed-error identity is checked numerically. Original frozen fit sources remain unchanged.
