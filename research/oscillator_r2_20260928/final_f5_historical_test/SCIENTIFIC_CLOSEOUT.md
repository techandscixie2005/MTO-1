# Final F5 scientific closeout

Independent review: **PASS**. Recorded 2026-09-29T09:05:28.309815+08:00. Recommend that root adopt the frozen F5 as the research champion for pooled raw-f R² in this completed campaign. Retain its exact recipe/checkpoints; no follow-up test query, refit or revised mixture follows from this result.

## Frozen comparison

| Predictor | Test pooled raw-f R² | SSE | MAE |
|---|---:|---:|---:|
| eta0 context | 0.455815109 | 103.193193599 | 0.017715985 |
| previous equal3 | 0.485101765 | 97.639596726 | 0.017265144 |
| frozen F5 | **0.508624899** | **93.178930153** | **0.016277304** |

F5 improves R² by **0.023523134** and reduces SSE by **4.5685%** against old3. Independent 2,000-draw paired bootstrap reproduction gives molecule 95% interval [0.009246645, 0.039133744], positive fraction 0.999; connectivity-group interval [0.009644333, 0.039002011], positive fraction 0.9995 (6,671 groups). All 6,686 molecules and 66,860 labels remain included.

## Gains and remaining errors

- Gains cover 4,052 molecules and nine states; 2,634 molecules worsen. Gross molecule gains are 9.215495 SSE and losses 4.754828, for net 4.460667. The largest improving molecule contributes 0.575088 SSE (12.9% of net); the ten largest contribute 34.0%. Group extrema coincide with the corresponding singleton molecules. This is distributed complementarity, not evidence that wider individual models or a specific architectural change caused the gain.
- S7 worsens by 0.761314 SSE. At the frozen TRAIN q90 cutoff 0.0549, below-threshold gain is 4.009199 and bright gain 0.451468 SSE. At TRAIN q99 = 0.2412, the 647 bright labels worsen by 1.199445 SSE; the remainder gains 5.660112. Mean signed error shifts from -0.000576617 to -0.002166925, so average underprediction increases despite lower total error.
- There are no test molecules in q2 <= 1e-5. The next fixed geometry bin has only five molecules and worsens by 1.518555 SSE (old3 9.252918 to F5 10.771473). Every other populated fixed q2 bin improves. The five-case stratum is descriptive and does not establish a geometry cause; no case was filtered. F5's worst single-molecule SSE is 6.055743 and its top ten account for 22.472003 SSE (24.12% of total), so substantial concentrated errors remain.

## Integrity, cost and interpretation

The independent audit verified the published-hash authorization before STARTED, exact protocol/freeze/evaluator/preflight/review binding, all 125 frozen source/artifact hashes, all five checkpoint and prediction hashes, raw FP64 truth/IDs/masks, complete disjoint splits, native G1/G3 E-times-FP64-trace arithmetic, fixed five-way mean, old3/eta0 historical reproduction, full metrics/slices/concentration, and unchanged physical GPU4 UUID/ECC. STARTED precedes both new passes and the successful terminal receipt; there are no failure markers. The pre-reviewed single-use evaluator and receipts support exactly two new selected checkpoint dataset passes, G1 epoch75 then G3 epoch174, while the three immutable eta arrays were reused. This audit performs no model inference or test evaluation rerun.

Deployment requires five neural forwards versus old3's three, a 5/3 model-count ratio rather than a measured latency ratio. The authorized execution measured 4.2594 s for G1 and 2.7301 s for G3 including batch input assembly/transfer, about 6.99 s total; it did not benchmark full-five deployment latency.

This historical test had already been reused, and F5 was selected after adaptive research on reused validation. The paired intervals describe this fixed comparison and omit model-selection/adaptation uncertainty; this is not untouched-holdout confirmation. The result supports a campaign research champion, with the stated cost, S7/bright-tail/geometry weaknesses and no causal architecture claim. The previously frozen validation-derived tail slices remain unchanged; these final slices use the actual TRAIN cutoffs bound by the additive erratum.

Sources: CANDIDATE_FREEZE.json, PROTOCOL.md, EXECUTION_AUTHORIZATION.json, STARTED.json, TEST_COMPLETE.json, TEST_RESULTS.json, and INDEPENDENT_COMPLETION_REVIEW.json. Exact hashes are in SCIENTIFIC_CLOSEOUT_PROVENANCE.json.
