# Frozen post-run geometry comparison protocol

2026-09-29. Validation-only postprocessing, separate from every active training source.

## Purpose and fixed inputs

Apply the reviewed Round09 geometry definition and train-derived cut points to the saved validation predictions of the completed seed23/37 and scratch readout studies. This tests whether their full-validation raw native-f error differences concentrate in previously observed geometric regimes. It performs no training, model inference, checkpoint selection, threshold search, sample filtering, or test access.

The immutable Round09 source is `../geometry_error_audit/geometry_error_metrics.json` (SHA-256 b088b23a5098149c5a127623a5a9315f6d4c979a501eeb3fd0af46285eaccbbd), with reviewed geometry code (SHA-256 cda3735529a57e83a6183ef82a85fc022022517404681bff0635a9bf91e54472). Reuse its FP64 centered unweighted covariance, q2=lambda2/lambda1 and q3=lambda3/lambda1, and degeneracy rule. q2 uses its existing six bins with boundaries 1e-5, .05941956341810567, .09867099135475822, .21033971729099188, .45009813312285296. Round09 records q3 as a ratio and its TRAIN quantiles at .001,.01,.10,.50 (1.3129937341794305e-11, 1.0273606733704654e-09, .03703241529916692, .1950348664308274), but did not define q3 bins. Retain q3 as a descriptive ratio and case diagnostic. q2 intervals are right-closed with a separately counted degenerate category. No new cut points or validation-derived bins.

Only the val indices of the pinned dataset.npz, raw_labels.npz, and identity_audit_v2.json may be read for geometry, IDs, masks and raw printed FP64 oscillator strengths. Lexically skip non-val identity rows and memmap only val geometry/raw-label slices. Require exact saved indices, IDs, truth, masks where present, shape, finiteness, and hashes. Every full-benchmark metric retains all 6,686 molecules/66,860 states.

## Predetermined predictors and selection labels

References: archived eta0 legacy-selected native f and the arithmetic mean of archived eta0/eta01/eta1 native f, called fixed equal3. The fixture mode uses these two alone to reproduce Round09 full and q2-bin SSE/counts before future outcomes are present.

Seed-completed mode requires both seed23 and seed37 FIT_COMPLETE receipts for exactly 100epochs/188100updates; scratch-completed mode requires both scratch native/MTO receipts; combined mode requires all four. Each family can close independently. All modes reject FAILED/INVALID markers and check reviewed source maps, receipt identity, selected epochs, fixed-metric hashes, saved-prediction hashes and selected raw-f SSE before consuming future predictions. Before reading any new prediction array, verify its hash against the matching terminal receipt. Use seed23/37 legacy-selected native-f individual arrays for the PRIMARY equal-seed11/23/37 ensemble. Also show seed23/37 raw-f-selected SECONDARY individuals and the predefined MIXED-SELECTION ensemble (legacy seed11 plus secondary seed23/37), clearly marked as such. Scratch native and scratch MTO use their independently minimum raw-f validation SSE selected arrays. Their native-minus-MTO paired difference is the primary scratch contrast. A prespecified secondary complementarity diagnostic is the 0.5 native + 0.5 MTO mean of those same selected raw-f arrays; this exact weight is fixed before outcomes and has no search. Do not substitute or tune weights. The seed ensemble and fixed equal3 each use exact equal arithmetic weights.

## Measures

For every predictor, compute pooled FP64 raw-f SSE/SST/R²/MAE/RMSE on the full validation benchmark, full per-state SSE and signed error, and fixed q2-bin molecule/label counts, SSE, SSE share, mean signed error and per-state SSE. Against eta0 and fixed equal3, report signed SSE differences, their absolute magnitudes, R² differences and per-bin/state delta SSE. The one-molecule case14562 and the fixed six molecules in Round09's two lowest q2 bins are concentration diagnostics only; never use their exclusion to change the benchmark.

Cross-check eta0/equal3 full and q2-bin SSE/counts against the immutable Round09 output at tight FP64 tolerance. Refuse absent/incomplete/failed FIT_COMPLETE records without reading active status, histories, logs, checkpoints or predictions. Explicit UTF-8 JSON/Markdown output. Preserve earlier results and source files.

## Commands

Fixture (allowed now): `python postrun_geometry.py --fixture`, writes `FIXTURE_VALIDATION.json` and `FIXTURE_VALIDATION.md`.
Seed family (after two seed receipts): `python postrun_geometry.py --seed-completed`, writes `SEED_RESULTS.json` and `SEED_REPORT.md`. Scratch family (after two scratch receipts): `python postrun_geometry.py --scratch-completed`, writes `SCRATCH_RESULTS.json` and `SCRATCH_REPORT.md`. Optional combined report after all four: `python postrun_geometry.py --completed`, writes `COMPLETED_RESULTS.json` and `COMPLETED_REPORT.md`. Completed modes require exact postprocessor independent review and all family-specific source/selection gates; a missing prerequisite exits PENDING without reading unfinished predictions or writing results. Existing completed outputs are preserved rather than overwritten.

