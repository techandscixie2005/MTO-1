# Independent Round09 scientific review — PASS

The fixed coupled AdamAMSGrad L2 coefficient 1e-4 failed the frozen dual +.003 allocation gate. Retain the published Round05 original control45 at raw-f R² .44716940136585204. This review supports closing the exact two-arm study; it authorizes no sweep, replacement optimizer, extension, seed fit or TEST release.

## Results and tradeoffs

| Policy | Zero decay | Coupled L2 | Coupled minus zero |
|---|---:|---:|---:|
| Earliest minimum validation SSE | .415283618 (epoch39) | .361496462 (epoch56) | −.053787156 |
| Fixed epoch60 | .407024298 | .348062884 | −.058961414 |

The candidate is −.085672939 below the retained reference. Both comparisons required +.003. All 66,860 printed raw-f labels, zeros, ten states and 6,686 validation molecules remain included. Selection considered all epochs0–60, with earliest ties retained. Selected-policy comparisons and aligned fixed60 comparisons are separate.

At selected checkpoints, coupled decay worsens pooled MAE (.020390 versus .018132), energy RMSE (.153195 versus .125263 eV), true-q90 RMSE (.108482 versus .097702), and true-q99 RMSE (.260109 versus .224858). S7 and S9 have lower SSE; the other eight states worsen. The same two states improve at fixed60. No uniform degradation claim is warranted.

Selected false-bright q99 SSE falls from16.063663 to4.937867, and its count from155 to39. Missed-bright count rises from418 to575 and its SSE from29.682902 to43.057371. The four bins exhaust every label and the whole SSE; membership depends on predictions, so these changes are descriptive decompositions rather than effects on fixed subgroups. Fixed TRAIN thresholds are .0549/.2406, with6,782/708 true-tail labels.

## Integrity and review scope

Both original registered workers completed60epochs/112,860updates, with one attempt each, no failure marker or recovery. QC's exact terminal receipt establishes process absence; an OS child exit code was not observed. Independent text/opaque-byte checks verified all360 frozen source files, exact authority/review/publication,61 history rows,60 order hashes, common initial-base contract, status/BEST/history consistency, original-mode dormant branches, WD0/1e-4, selection and checkpoint/export/prediction hashes. Access ledgers match the frozen TRAIN/validation indices and record zero TEST numeric rows.

Science executed the reviewed CPU analysis once. It verified the ordered135-tensor optimizer roster, options/default dispatch, moment shapes/steps/finiteness and AMSGrad maxima, checkpoint RNG categories, selected/export tensor equality, config/stats/provenance, and identical truth/masks across four saved validation sets. This reviewer read that source and its aggregate output, rehashed all32 referenced inputs as opaque bytes, checked pooled/state/tail/bin/energy arithmetic and gate deltas, and did not decode checkpoint or prediction arrays, construct a model, run inference or repeat reductions. Stored export-buffer fingerprints were checked as metadata; preparation established reconstruction parity.

## Interpretation limits

The2,000 paired bootstrap draws resample5,656 audited validation identity groups with seed20260930 and recomputed pooled SST. Coupled-minus-zero intervals are [−.096794,−.009862] selected and [−.090203,−.028389] fixed60. Both are below zero conditional on these trained checkpoints and reused validation. They exclude checkpoint-selection, training-seed and fresh-generalization uncertainty. No paired interval is available against the aggregate-only retained reference.

This was coupled decay applied internally by Adam after task-gradient clipping and before moments, including all original trainable biases, radial parameters and energy offsets. It was not AdamW or an added pre-clip loss. At selected checkpoints, parameter norm is65.080 versus841.656, minimum absolute radial beta .019233 versus2.195434, and energy-offset range3.434519–4.975100 versus5.472208–7.514449. These are real treatment-associated differences; they do not isolate which parameter family or adaptive-optimizer effect caused the accuracy loss. Effective-gradient norms are not adaptive update norms. TRAIN diagnostics are pre-update trajectory aggregates, not fixed-checkpoint generalization-gap estimates. Late validation decline does not prove overfitting.

The result is evidence against this exact all-parameter coefficient at this budget, not against regularization in general. Repeated declared seed11 controls vary across rounds; they are not independent-seed confirmations or a causal explanation of this contrast. The v2 partition is disjoint under audited conservative rules but uses historically exposed data; its sealed TEST includes5,989 oldTRAIN/361 oldvalidation/336 oldTEST rows. No independent external, seed or TEST confirmation is claimed. The .60 goal remains unmet.

## Persistent evidence

- Source review: ../TERMINAL_ANALYSIS_SOURCE_REVIEW.json
- Independent metadata: INDEPENDENT_TERMINAL_METADATA.json
- Independent aggregate checks: INDEPENDENT_RESULT_CHECKS.json
- Science outputs: ROUND09_RESULTS.json and ANALYSIS_RECEIPT.json
- Narrative/commands: ROUND09_REPORT.md and ROUND09_REPRODUCE.md

Final closeout manifest binds exact bytes and the science terminal inventory separately. Root owns the closeout decision and publisher owns D-first archival publication; neither creates new numerical authority.
