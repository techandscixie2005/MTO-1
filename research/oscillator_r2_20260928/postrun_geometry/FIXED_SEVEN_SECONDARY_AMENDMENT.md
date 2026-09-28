# Fixed seven-model secondary amendment

Recorded at: 2026-09-29T04:41:20.245408+08:00

**Status:** Post-launch, pre-seed-outcome specification, authorized by root. Design/protocol only; no implementation or execution here. Existing training/evaluator versions remain sealed. This amendment follows inspection of the completed chan64/fixed-five validation results; seed23/37 outcomes have not been read for this decision. It is exploratory, not an independent preregistered confirmation.

## One additional candidate

Let F5 be the already frozen equal native-f mean of eta0, eta01, eta1, G1 epoch75, and G3 epoch174, using the exact component array hashes in the companion provenance. Define exactly

`F7 = (f_eta0 + f_eta01 + f_eta1 + f_G1 + f_G3 + f_seed23_legacy + f_seed37_legacy) / 7`.

Each individual model has weight 1/7; old eta0 appears once. Equivalently, `(5*F5 + f_seed23_legacy + f_seed37_legacy)/7`. Do not average F5 and seed3 as two blocks: that changes weights and duplicates eta0. The two future seed components must be the prescribed LEGACY-JOINT-selected checkpoints, never raw-f-selected replacements. No raw-selected seven-way variant, subset/weight search, additional fit/inference/test, changed split, or training change is permitted.

The original seed3 `(f_eta0 + f_seed23_legacy + f_seed37_legacy)/3` remains the seed study primary ensemble. Its separately labeled mixed-selection secondary remains unchanged. Compare F7 to both F5 and that primary seed3, retaining the full 6,686 molecules / 66,860 transitions. These are the only added candidate contrasts; existing study reports remain intact.

## Future saved-input gates

Consume future arrays only after BOTH seed workers have genuine valid `FIT_COMPLETE.json`: correct seed23/37 identity, 100 epochs, 188,100 steps, no FAILED/INVALID marker, and valid legacy-selected epochs 1..100. Require source/code maps to equal the pinned reviewed implementation, pinned preflight and protocol; require `test_batches==0`. Bind `FIXED_CHECKPOINT_METRICS.json` to the terminal hash and its legacy epoch to `best_legacy_epoch`. Use only `runs/seed23/val_best_legacy.npz` and `runs/seed37/val_best_legacy.npz`, whose hashes equal terminal `val_prediction_hashes.best_legacy`; record corresponding legacy checkpoint hashes from the completion audit. Honor the existing fixed-checkpoint numeric tolerance without relabeling joint selection as raw-f selection. A missing or inconsistent receipt/array is a stop, not permission to replay.

Verify all seven native-f arrays are finite FP64 for arithmetic and share exact validation indices, molecule IDs, state order, raw printed FP64 f truth and masks with the frozen five. Raw E must never enter predictions. The existing five component hashes are fixed now; future seed hashes must be recorded from completed receipts before combination. Arrays/checkpoints remain server-only. Seed analysis does not depend on scratch completion.

## Required report

Report pooled raw-f FP64 SSE/SST/R², RMSE and MAE for F7, F5 and primary seed3; absolute SSE and signed changes accompany shares. For F7-minus-F5 and F7-minus-seed3, use paired 2,000-draw molecule and connectivity-group bootstrap, seed 20260929, shared unit resamples for each candidate/reference pair, recomputing pooled SST per resample; report 2.5/50/97.5 percentiles and positive fraction. Preserve each molecule's ten states. These intervals are descriptive conditional on validation selection, not multiplicity-adjusted evidence.

Report physical S1..S10 SSE changes, below-q90 and q90/q99-and-above SSE/counts with frozen thresholds 0.0546/0.2377 (ties included); largest positive and negative molecule/group changes, top 10 positive changes, gross improvement/deterioration, number of improved molecules, ID 14562, and the frozen Round09 q2 bins/six-case concentration. Use exactly the existing TRAIN-derived q2 boundaries plus fixed near-linear threshold; no fresh bins or filtered headline. Rare-case concentration and tail tradeoffs are diagnostics, not automatic rejection rules. If net gain is nonpositive, do not present unstable percentages as positive gain shares.

Inference requires seven model forwards versus five for F5 and three for seed3. G1/G3 have larger readouts; counts are not calibrated latency ratios. The hypothesis is additional error complementarity from independently initialized seeds. A gain cannot establish width, backbone, or readout causality, and a loss does not reject seed diversity generally. The same validation set selected checkpoints and informed research; the historically reused test is not reopened. Material gains may motivate a later frozen-protocol confirmation decision, not automatic promotion.

## Supplemental scratch reference only

After BOTH scratch arms satisfy their existing terminal/integrity gates, permit frozen F5 as a supplemental reference for each prescribed raw-f-selected scratch arm and the already authorized fixed 0.5/0.5 scratch mean. Preserve native-minus-MTO as the primary contrast and the scratch mean as secondary. Reuse the same full-benchmark metric, paired uncertainty, state/tail/concentration conventions; label the F5 comparisons supplemental and exploratory. Do not add scratch to F7/F5, alter scratch selection, or require seed completion for scratch reporting.

## Scientific assessment

No material objection: the arithmetic is cheap and the fixed-five gain makes this a targeted complementarity question. Limitations are the post-individual-result choice, selected/reused validation, unequal inference budgets, and shared training data. This amendment authorizes no execution by itself; any later utility amendment requires independent source/hash review. No current sealed source was edited and no active outcome was inspected while writing it.
