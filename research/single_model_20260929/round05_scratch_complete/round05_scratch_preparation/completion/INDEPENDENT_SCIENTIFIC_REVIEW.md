# Round05 independent scientific closeout

**PASS for the completed evidence. No modified arm passes the predeclared continuation gate.** All four fresh models completed 60 epochs and 112,860 updates. Retain control epoch 45 as the current QM9S v2 validation-selected reference. The R² 0.60 objective remains unmet; no TEST result or independent-seed confirmation exists for this round.

## Matched results

Each value uses all 66,860 validation labels, including zeros. Selection is earliest minimum pooled raw-f SSE over epochs 0–60.

| Model | Selected epoch | Selected raw-f R² | Difference from selected control | Fixed-60 R² |
|---|---:|---:|---:|---:|
| Original control | 45 | 0.447169401 | — | 0.402691932 |
| Right-F | 41 | 0.415020369 | -0.032149033 | 0.401177943 |
| Raw-M decorrelation | 23 | 0.412399193 | -0.034770209 | 0.392013476 |
| Both | 20 | 0.424966054 | -0.022203348 | 0.402391091 |

Every modified model also trails control at epoch 60. The +0.003 selected-checkpoint gate fails; these results do not support extending these four recipes or allocating confirmation seeds to a modified arm. This closes the bounded factorial, not every possible equivariant transformation or regularizer.

Control's one-checkpoint geometry predictor is:

`/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`

SHA256: `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`.

The checkpoint embeds the selected model, architecture settings and new TRAIN statistics. Its tensors exactly match the selected model in the reviewed CPU terminal analysis. The previously reviewed loader requires only geometry-derived inputs and this checkpoint; no QC labels, old weights, caches or prediction averaging are used. No new inference was needed for closeout.

## Tradeoffs and interpretation

| Selected model | MAE | TRAIN-q90 RMSE | TRAIN-q99 RMSE | Energy RMSE, eV |
|---|---:|---:|---:|---:|
| Control | 0.0174631 | 0.0980282 | 0.2249820 | 0.1229826 |
| Right-F | 0.0179421 | 0.0970732 | 0.2175896 | 0.1270609 |
| Decorrelation | 0.0182967 | 0.1029280 | 0.2403345 | 0.1300376 |
| Both | 0.0188370 | 0.1008308 | 0.2354149 | 0.1385316 |

Tail cuts .0549/.2406 were frozen from TRAIN and include 6,782/708 validation labels. Right-F improves these selected-checkpoint tail RMSEs but loses pooled R² and MAE. Its false-bright-bin SSE is 19.2076 versus control's 12.3345. Decorrelation and both reduce that bin to 9.4906 and 9.2606 but lose through errors elsewhere. All four brightness bins partition every label; no exclusions or alternate checkpoint selections were introduced. Per-state results also vary: control's selected R² ranges from .8354 for S1 to .1588 for S10; the modified arms improve some states and worsen others. The full ten-state comparisons remain in ROUND05_RESULTS.json and the round report.

The largest 1% of absolute errors contribute 55.58% of control's selected SSE. This is an observed error concentration, not proof of bad labels or a reason to remove molecules. All labels remain in evaluation.

The modules are active. At epoch 60, TRAIN mean relative F updates are about .536 and .571 in F/both. Decorrelation's trajectory mean raw-M squared cosine is .4014 versus control's .5093. These are learned-feature diagnostics, not physical wavefunction orthogonality or electronic phase covariance.

Control epoch 45→60 lowers TRAIN trajectory base loss .08980→.06446 while validation trace loss rises .14343→.15306 and pooled R² falls .44717→.40269; energy RMSE improves slightly. This is consistent with a generalization limitation. It does not isolate a causal mechanism: TRAIN logs aggregate changing pre-update minibatch models, validation uses one completed checkpoint, and only one seed was run.

The aligned epoch-60 factorial R² interaction is +.0118916. It describes the joint modification being less harmful than the sum of separate changes in this run; both still trails control. The +.0447159 arithmetic contrast between separately selected checkpoints uses different epochs and is a pipeline-level description. Neither is evidence of a physical interaction or grounds for promotion.

## Uncertainty and historical exposure

The analysis resamples 5,656 audited validation identity components, retaining all states and linked molecules, for 2,000 paired bootstrap draws. Selected-checkpoint descriptive intervals for R² differences versus control are F [-.07860,.00754], decorrelation [-.06014,-.01073], both [-.05056,.00123]. All fixed-60 intervals cross zero. These intervals condition on the already trained and selected checkpoints. They are not corrected for validation reuse/checkpoint selection, do not measure seed variability, and are not fresh generalization confidence intervals. The point-estimate allocation gate remains unchanged.

The v2 split is a separately versioned group-disjoint partition under audited conservative rules, with fresh initialization and TRAIN-only normalization. It is historically exposed data: its TEST contains 5,989 old TRAIN, 361 old validation and 336 old TEST molecules. No v2 TEST target was decoded or scored. The old eta0/calibrated scores are different-partition historical evidence and cannot establish a gain or regression relative to this .44717 reference.

## Integrity and scope of independent review

- All 82 frozen source/dependency hashes remain unchanged. Each arm has one registered original attempt, all 61 epoch records, the same 60 order hashes, 1,881 batches per epoch, and matching completion/checkpoint hashes. Owned monitoring observed terminal workers absent; an OS exit code is not inferred from that observation.
- The independent metadata audit reconciled pooled/per-state/bin counts and error arithmetic across all 244 history records, plus earliest-best selection, access ledgers and checkpoint hashes. Checkpoints and saved arrays were opaque-hashed by this reviewer, without tensor/array decoding.
- The reviewed science analysis checked checkpoint optimizer/RNG/history metadata and exact selected-model/export tensors, then recomputed all selected/fixed-60 metrics from eight committed saved validation arrays. Their truth/masks matched exactly. It performed no model construction, forward pass or raw-dataset target read.
- The independent result audit verified all 54 analysis source/input/output bindings and reconciled the reported metrics, trajectory snapshots, gate and interaction arithmetic against the independently audited histories. Bootstrap formulas and grouping were source-reviewed; the bootstrap was not repeated by this reviewer.

Supporting receipts: INDEPENDENT_TERMINAL_INTEGRITY.json (`417950cf…`), TERMINAL_ANALYSIS_SOURCE_REVIEW.json (`b1ffea47…`), completion/ANALYSIS_RECEIPT.json, completion/INDEPENDENT_RESULT_CHECKS.json (`8ebe5538…`). No scientific implementation, fitting, inference or TEST evaluation was added during review.

Root owns the next scientific decision. A fresh unchanged-PSD objective comparison could ask a narrower new question than adding another readout: the audited raw-f scratch study changed to a direct scalar-f head, while original-PSD raw-f studies were warm starts or frozen-base corrections. That history distinction does not predict success or authorize a new experiment. Any follow-on needs its own fixed protocol and matched control; no unchanged failed recipe should be silently repeated.
