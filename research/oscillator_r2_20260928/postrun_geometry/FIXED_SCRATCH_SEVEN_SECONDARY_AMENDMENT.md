# Final neural-batch mixture amendment: fixed scratch seven

Recorded at: 2026-09-29T07:25:14.365185+08:00

**Status:** root-authorized design only. Post-native-completion-result, pre-completed-MTO/paired-scratch-outcome review; prior interim information was already available. This is explicitly exploratory. No model inference, test access, fitting, training change or utility implementation is authorized by this document. Terminal audits take priority; any later implementation must be isolated, independently reviewed and source-hash sealed.

## Exactly one new mixture

Define `F7_scratch = (5*F5 + scratch_MTO_rawselected + scratch_native_rawselected)/7`, with F5 the existing immutable equal mean of eta0, eta01, eta1, G1 epoch 75 and G3 epoch 174. All seven INDIVIDUAL components have weight 1/7; no averaging of differently sized ensemble blocks. The two scratch components use their already prescribed minimum raw-native-f validation SSE checkpoint over epochs 0..100, earliest exact tie. Native epoch 21 is already completed; its selected checkpoint/array/receipt are pinned in provenance. No alternative joint-selected scratch mixture is permitted.

Compare once against F5. Also compare directly against the existing fixed `F7_seed_legacy=(5*F5+seed23_legacy+seed37_legacy)/7` when both seed completions are available. The seed recipe/selection remains unchanged. Do not substitute raw-f-selected seeds, form all-nine, tune weights, try subsets, or average the two sevens.

## Frozen roster for this neural batch

This is the **final additional neural mixture** in the current batch. Existing individual-model reports remain permitted. The complete fixed mixture roster is:

1. Original equal3: eta0/eta01/eta1, one-third each.
2. Chan64 equal2: G1/G3, one-half each.
3. Frozen F5: eta0/eta01/eta1/G1/G3, one-fifth each.
4. Seed3 PRIMARY: archived eta0(seed11 legacy) plus seed23/37 legacy-joint selections, one-third each.
5. Seed3 MIXED-SELECTION SECONDARY: archived eta0 legacy plus seed23/37 raw-f selections, one-third each; old eta0 is not relabeled raw-f-optimal.
6. Scratch equal2 SECONDARY: native/MTO raw-f-selected components, one-half each; native-minus-MTO remains the primary scratch contrast.
7. Existing F7_seed_legacy as defined above, all seven weights 1/7.
8. New F7_scratch as defined above, all seven weights 1/7.

The separately approved descriptor study's standalone D and fixed 0.5D+0.5F5 are a distinct non-neural probe and remain unchanged. No further neural mixtures are authorized. Any later test proposal requires root's explicit candidate/recipe freeze and independent review first; this roster does not authorize a test query. Historical test reuse prevents a fresh-holdout confirmation claim from another query.

## Completed-artifact gates

Before any scratch selected array is consumed for this new mixture, BOTH scratch arms must have genuine FIT_COMPLETE receipts: correct arm identity, 100 epochs/188100 steps, test_batches=0, no FAILED/INVALID markers, reviewed source/code/preflight/protocol maps unchanged, and valid raw-f-selected epochs 0..100. Bind selected checkpoint bytes, versioned aliases, saved prediction hashes and fixed-checkpoint metrics/selected epoch to those receipts, using the already reviewed sidecar's numeric tolerances. Native's now-frozen pins must match; MTO's future pins must be collected from its completed receipt before combination. Missing artifacts or mismatches stop analysis and authorize no replay.

The optional F7_scratch-minus-F7_seed_legacy contrast additionally requires BOTH seed workers' genuine 100-epoch/188100-step terminals and unchanged prescribed legacy selection/receipt/checkpoint/array gates. If seeds are unavailable, label this contrast pending; do not replace it or depend on its future score to report the F5 contrast. Seed-only original analyses do not wait for scratch. Existing primary results and original utility outputs are not overwritten.

All component arrays must have exact fixed validation indices, molecule IDs, ten-state order, raw printed FP64 f labels and masks. Predictions must be finite native f; neither oracle E nor target-derived corrections enter the arithmetic. Retain all 6686 molecules/66860 transitions, and keep all prediction arrays/checkpoints server-only. F5 component hashes and existing protocol/review pins are fixed by the companion provenance.

## Fixed reporting and interpretation

Report full raw-f FP64 SSE/SST/R², MAE and RMSE for F7_scratch and its available fixed references. For each prescribed contrast use 2000 paired whole-molecule and connectivity-group draws, seed 20260929, original validation molecule order and sorted connectivity-key group order, keeping all ten states together and recomputing pooled SST. Report percentile 2.5/50/97.5 and positive fraction. Direct paired differences matter; do not infer a difference by comparing separate CI endpoints.

Report S1..S10 SSE, below-q90/q90-and-above/q99-and-above at frozen thresholds 0.0546/0.2377 with ties included; signed errors; largest positive/negative molecule and group contributions, gross gains/losses, top 10 positive contributions, ID 14562 and frozen Round09 q2 bins/six-case concentration. Absolute SSE accompanies shares; nonpositive net gain has no positive gain-share headline. No evaluation case is filtered or downweighted. Concentration and bright-tail losses are diagnostics, not hard rejection rules.

F7_scratch requires seven model forwards versus five for F5 and seven for F7_seed_legacy. Readout sizes and schedules differ, so counts do not establish equal latency or matched training cost. Report model bytes/time only from available measurements, with limits. A gain supports error complementarity for these frozen models, not pure architecture/width causality: scratch/native/MTO and legacy seeds differ in objective, selection, schedule and initialization history. Weak single-model R² alone does not rule out useful averaging. A null result does not reject nonlinear invariants or a backbone family generally.

No material scientific objection to this inexpensive fixed saved-array question. Its adaptive timing adds selection uncertainty; bootstrap intervals remain descriptive under repeated validation reuse. No promotion, fresh confirmation or test authorization follows automatically.
