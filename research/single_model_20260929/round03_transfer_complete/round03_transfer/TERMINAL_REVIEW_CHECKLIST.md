# Round03 bounded independent terminal review

Prepared while source training was still active. This is a checklist, not a completion claim. Science owns the affine commands; monitor owns process/resource monitoring. The reviewer runs no model inference, fitting or new raw-label inspection.

## Immutable contract

- Manifest: `93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37`, all69 referenced files.
- Prelaunch review: `a6807aae55c59f254f2048b1d696a754f130075e32ef4763dfa1c94f4fb1c04c`.
- Preparation publication: `1b34e839335177e1fd582944549be4f181dee3c0`, parent `bdc28c351b941a58dae49f833284e13cf5fdc46b`; publication receipt SHA `1b28241ea82319087cc6681c0d36caf18a837ee8fc3b84fc6287429d0174df37`.
- Owned source: PID1174422, GPU1, launch identity `round03_source33_1790707949921845597`. Treat process identity/exit as monitor evidence, never infer completion from elapsed time or an agent status.

## Source terminal

1. Rehash the current manifest/closure and bind review, publication and launch receipts. Confirm no scientific settings changed.
2. Require genuine FIT_COMPLETE at33 and the exact manifest, with no unresolved failure. Require monitor evidence that the original process has exited before affine execution. Distinguish observed terminal state from an OS exit code if none is retained.
3. Check source_final.pt and last.pt byte hashes against completion receipt. Inspect CPU checkpoint metadata/state only: fixed epoch33 selection, config/statistics, fit-index and initial tensor hash, source-final tensors versus last state, optimizer/RNG presence, completed history and49,665 updates.
4. History must contain exactly epochs1..33, each1,505 batches at lr0.001. Regenerate all33 NumPy seed11 permutations using fit index metadata and match order hashes. No best/checkpoint search, validation metrics or held-out target path belongs in source fitting.
5. Reuse guarded initialization/fit-access preflight and STATISTICS_AUDIT. Do not repeat fitting or decode original target arrays. Report source size/update count/quality as a confounder.

## Affine ordering and provenance

1. Only science executes fit/export/evaluate under reviewed locks. Source terminal/exit must precede that execution.
2. Both maps use the same24,071 held rows and all240,710 valid printed-f labels. Verify index/array hashes and aggregate counts against the frozen split contract; zeros remain valid. Do not infer missingness from predictions.
3. COEFFICIENTS_FROZEN must contain both pairs and be bound to the source checkpoint, original full baseline and calibration array SHA. Both pairs must predate every outer-validation attempt. Inspect attempt logs/timestamps and immutable hashes; timestamps alone do not replace reviewed execution order.
4. No refit or coefficient selection after outer results. An interrupted export may only reuse the same receipt and predictions. The fit is the prescribed unconstrained FP64 affine procedure; clipping occurs only at application.
5. Each composite checkpoint must contain exactly the original full-baseline tensors/config/statistics plus its one coefficient pair. Verify CPU tensor equality and provenance. No temporary-source dependency, historical-calibration composition, averaging or QC-input requirement at deployment. Reuse recorded geometry replay evidence; perform no duplicate inference.

## Metrics and decision

1. Verify aggregate reports bind exact source/checkpoint/array/code hashes. Validation retains6,686 molecules/66,860 labels. Native R² must replay .4052941183410983 and historical fixed calibration approximately .418119238799 within1e-7.
2. Check pooled R² equals1-SSE/SST, RMSE equals sqrt(SSE/count), per-state counts and SSE sum to pooled totals, and all map counts/SST agree. State SST values do not sum to pooled SST because their centering differs.
3. Check fixed TRAIN bright cuts .0549/.2412, nested tail counts/SSE, finite aggregate values, unchanged energy metrics and individual map coefficients. Report pooled/state/tail tradeoffs alongside any gate result.
4. Recompute the arithmetic gate from recorded metrics: heldout map minus in-sample map>=.003 AND heldout map minus native anchor>=.003. This allocates nonlinear preparation only, subject to root assessment. It is not seed promotion or a best-model claim; historical calibrated .41811924 remains an exposed comparator.
5. Describe source quality, training-size mismatch, reused outer validation, missing fresh holdout, and possible E/A-versus-calibrated-f inconsistency. No test scoring.

## Closeout

Science supplies results and interpretation; the reviewer supplies integrity/scientific findings; root makes the next decision. Monitor first downloads all lightweight records to D:\MTO\archives, verifies them, then inspects/stages/commits/pushes. Checkpoints, optimizer state, index/data/prediction arrays and credentials remain server-only. An incomplete gate remains incomplete; no success is inferred.
