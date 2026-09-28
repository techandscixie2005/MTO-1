# Fixed scratch mean: secondary analysis amendment

Timestamp: 2026-09-29T03:42:32.728714+08:00. Root explicitly authorized this amendment after scratch launch and before any scratch outcome review by root/reviewer. This is an additive analysis protocol outside sealed training sources. No new fitting, checkpoint selection, model inference, training change or test access.

Once BOTH scratch arms have valid FIT_COMPLETE receipts and verified selected prediction hashes, add exactly:

    scratch_fixed_equal_mean = 0.5 * native_raw_f_selected + 0.5 * mto_raw_f_selected

Each component is the already prescribed minimum pooled raw-f validation SSE checkpoint from epochs0..100. Require exact shared validation IDs, indices, FP64 truths and masks where saved. No per-state weights, subset search, clipping, refitting or subsequent mixture adjustment. Preserve all6,686 molecules and all66,860 labels.

The original selected native-minus-MTO comparison remains primary. Label the fixed mean POST-LAUNCH/PRE-OUTCOME SECONDARY and report full raw-f SSE/R2/MAE/RMSE, signed SSE/R2 differences versus each component, archived eta0 and fixed equal3, and unchanged Round09 q2-bin/state/case concentration diagnostics. Show absolute SSE beside shares. Q3 remains descriptive; no new bins. Do not create excluded-case headline scores.

Hypothesis: the two readout families may make complementary errors even if neither is individually best. The mean's arithmetic and model-selection provenance are fixed now; comparison uses the same reused validation data and remains exploratory, including any subsequent uncertainty calculation. A gain is evidence for further confirmation, not proof of generalization or promotion. Independent training-seed confirmation, if later justified by material absolute benefit and inference cost, requires a separate frozen protocol. Historically reused test data cannot supply fresh confirmation through another query.

The utility must gate this predictor on completed scratch inputs only; seed-family-only mode must neither require nor read scratch predictions. Completed reports must record this amendment's exact hash. No new experiment is authorized.
