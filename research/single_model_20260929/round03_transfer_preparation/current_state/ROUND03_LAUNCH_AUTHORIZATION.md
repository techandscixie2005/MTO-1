# Round03 conditional execution authorization

Root authorizes the fixed source-generation and affine-transfer stage defined in ROUND02_DECISION.md once the gates below pass. No extra user or root permission is needed after those gates. This authorization does not cover a nonlinear readout fit, any source checkpoint search, or historical-test scoring.

## Fixed scope

- Reproduce the independently audited group-preserving96284/24071 internal split from original TRAIN, seed20260930, retaining unresolved groups in source fitting. Match the recorded identity/index hashes. No original TRAIN molecule is dropped; outer validation/test stay unchanged.
- Build a fresh original eta0 MTO source using seed11, without learned weights or target-derived initialization from the full-TRAIN model. Recompute E_state_mean and every target-derived normalization from its96284 fitting molecules only.
- Train exactly33 epochs with original LE+Ls, Adam AMSGrad lr0.001/batch64/clip5/weight-decay0/FP32 and pinned original optimizer defaults. No early stopping, held-out/calibration/outer-validation/test checkpoint selection, or scheduler reduction. Save atomic resumable states. Use the fixed epoch33 source.
- Generate predictions on the same24071 reserved molecules from that source and from the existing full baseline. Fit exactly one unconstrained pooled affine least-squares map for each prediction source in FP64, retaining every valid raw printed-f label. Apply max(0,a*f+b) identically. Report invalid/nonfinite/zero-variance fitting failures; no silent replacement or exclusions.
- Freeze both coefficient pairs before evaluating the resulting predictors on outer validation. BOTH predictors use the unchanged full baseline's NATIVE f. The historical calibration is a separate comparator, never an input to another affine map. No averaging and no temporary source at deployment.
- Evaluate native/full historical-calibration references and both new maps on all valid raw-f labels, including per-state and fixed TRAIN q90/q99 tail metrics. Document source quality and training-size mismatch. No test scoring.
- The advancement gate in ROUND02_DECISION.md allocates only nonlinear-readout preparation: at least0.003 native validation R² above BOTH the in-sample-trained map and native anchor. Approaching the historical validation-fit map is not a new-best claim or an independent-seed promotion.

## Launch gates

1. Science writes an exact protocol/config/source closure matching this scope, including data-access boundaries, split generation, initialization/statistics, source fitting, moment fitting, final geometry-only single-checkpoint export, resume and monitoring commands.
2. History independently reviews the exact frozen closure and meaningful preflight evidence. Verify label-clean initialization, group/index identities, subset-only statistics and training access, fixed33 selection, and no deployment dependence on the temporary source or QC labels. Reuse established checks where valid.
3. Monitor finishes round02 archive/publication first. Then FIRST download and verify round03 lightweight preparation records to D:\MTO\archives\; inspect byte-exact staged records; commit/push as a descendant of the verified research branch; issue a receipt bound to source/review/archive identities. No weights/data/prediction arrays/caches/credentials.
4. Immediately admit a healthy free GPU, hold the shared lock and register a distinct owned source job. Existing four-hour monitoring supports train-only progress and33epochs; preserve unrelated jobs and old source closures.

Science may then launch the single source fit and carry out the frozen affine stage after normal completion. Failures must be diagnosed on the owned job, with resumable states preserved. Fix/review failed gates before launch; do not silently alter scientific settings.

Estimated source cost is roughly60–75 GPU minutes, plus small inference and moment-fit costs. This repeats the original model recipe as a controlled source of held-out predictions, not as a proposed architecture improvement. A single internal holdout is not full cross-fitting or fresh outer confirmation. Final results still require independent review, a root next decision and archive-first publication.
