# Round04 decision and publication to main

Decision: **retain the calibrated eta0 single-model incumbent; close this fixed scalar-flexibility experiment without extension, refit or seed promotion.** The user's latest instruction authorizes publishing the reviewed results and updating `README.md` on `main`.

## Evidence and interpretation

Round04 completed exactly two fixed FP64 coefficient solves, immutable exports and one predeclared comparison on the existing validation cache. All 6,686 validation molecules and 66,860 valid raw-f labels were included. No test scoring, new benchmark evaluation, model averaging or checkpoint search occurred.

| Predictor | Pooled validation raw-f R² |
|---|---:|
| Native eta0 | 0.405294124 |
| Historical validation-fitted affine incumbent | 0.418119244 |
| In-sample affine control | 0.402024351 |
| Held-out-source affine control | 0.416374515 |
| In-sample four-coefficient map | 0.398331068 |
| Held-out-source four-coefficient map | 0.411775449 |

The new maps lose 0.00369328 and 0.00459907 R² against their respective affine controls. Neither meets the frozen requirement of at least +0.003 over both its matched affine and the incumbent. The held-out-source contrast of +0.01344438 between the new maps does not establish a benefit from added scalar flexibility. Descriptive paired-molecule intervals include zero; do not claim statistically established degradation or a causal mechanism.

The held-out-source map improves only S5 against its affine control. Its true q90/q99 errors and false-bright SSE are worse despite fewer false-bright predictions. Retain all state and tail comparisons in the report. This result closes the tested fixed-knot recipe; it does not prove that every nonlinear model or every transition representation must fail.

Integrity review PASS: `INDEPENDENT_TERMINAL_INTEGRITY.json`, SHA256 `99eee5a26d300ca92a4a4c588f61fa2d7c0e4710aba9c346e0acace124f7d058`. It verifies frozen source, terminal inputs, private artifact hashes, process/authorization records, order, aggregate arithmetic and promotion logic without re-evaluating arrays.

Independent scientific review PASS: `INDEPENDENT_TERMINAL_REVIEW.json`, SHA256 `011bca73a0376c7ce150562c76bd8d740c15b66bdcef50746f73b51b5eada8eb`. The seven-member review manifest is `9cdcf1d9a2bcca17881b8a3150f1aa1b84191817dcefce8be5b469ca37738bc6`. Root accepts its evidence limits and no-promotion conclusion.

Completed report: `ROUND04_REPORT.md`, SHA256 `4cdeafe40f86145b9020debf2d76d32a0771ddb7e914adc73a2df16e96a8e010`. Terminal manifest: `8c5defec95b7eb2d06e16cb960b5d82f8981aae9ec153e2519dfc47ef8b3a396`. Scientific source manifest remains `79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f`.

## Retained recipe and limitations

The retained predictor is one original eta0 seed11 epoch33 checkpoint plus `max(0, 0.8511830211044088 * native_f + 0.003725185373211049)`. It requires geometry and the embedded model state, not QC labels or a second checkpoint. E/A remain auxiliary and generally do not reconstruct the calibrated f.

Server checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`, SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.

Full-validation refit/replay R² is 0.4181192453; historical calibration OOF R² is about 0.416658 and still inherits prior checkpoint selection. Reused historical-test R² is 0.459667233057. These are different evidence types and must be labeled separately. The target of 0.60 has not been achieved, and no independently fresh test confirmation exists. Commit `69bf39f`'s five-model ensemble is ineligible for the current single-model objective.

The proposed new QM9S group-disjoint partition has not been materialized or validated. No fresh model has trained on it. New external data compatibility and overlap remain unverified. Preserve that distinction in the README; do not present pending research as a result.

## Authorized publication

Publish the completed reviewed Round04 records, root decision and approved README text in one coherent descendant of research commit `74810c5ad8a984fbf119dccd2e22336a112de7f5`. The verified current `main` is `69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e`, an ancestor of that research tip. Preserve all existing history and useful README setup/project documentation.

FIRST package the exact lightweight records and approved README bytes on the server, download to `D:/MTO/archives/single_model_20260929/round04_fonly_complete`, and verify hashes. THEN inspect the exact staged tree, commit, and atomically non-force fast-forward `main` and `codex/single-model-20260929` to the reviewed final commit. Recheck both remote tips before pushing; reconcile any intervening changes rather than forcing. Verify both remote refs and README/report blob contents after publication. No checkpoint, learned coefficient tensor/array, optimizer state, raw dataset, identity/split array, prediction cache or credential may enter the new commit.

The separate-index workflow must leave the dirty local working directory and publication repository's existing HEAD/worktree untouched. Exclude unfinished split-builder and QC-design drafts from the completed-results archive. After publication, update current operational handoffs and receipts without modifying sealed historical snapshots. No new experiment is part of this publication task; the continuing research objective and its reviewed-pilot requirements remain recorded in `RESEARCH_RESUMPTION_20260930.md`.
