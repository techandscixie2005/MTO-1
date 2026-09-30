# Independent Round04 terminal and scientific review

**PASS for completed-result publication; neither new map is promoted.** The retained single-checkpoint incumbent is the historical validation affine eta0 recipe. R² 0.60 has not been achieved. This review used frozen source, terminal receipts, aggregate arithmetic and file hashes. It did not decode prediction/target arrays, load models, repeat a solve or run inference.

## Integrity

The independent audit rehashed all 51 frozen source/dependency entries and all 26 terminal-manifest inputs. The execution authorization, independent preparation review and preparation publication are bound to the same source manifest. There are exactly two per-arm solve-start/solved records, one fit/export process, one evaluation process and one validation attempt. Both registered CPU wrappers exited zero, retained the same authorization and recorded child identities. Both immutable coefficient hashes and composite-checkpoint hashes match their receipts. Each solve preceded the common coefficient freeze, which preceded validation. No source changed.

The private checkpoint schemas, coefficient equality and complete backbone tensor/buffer equality are covered by the unchanged reviewed execution assertions and successful export/replay receipts. This independent terminal pass verifies the corresponding files' hashes without loading the private tensors again. The first 64 internal calibration geometries passed native/cache tolerance (maximum 8.7763e-7) and scalar-map arithmetic tolerance (2.2204e-16). This is parity evidence; batching-related numerical variation was not separately isolated experimentally.

All six reported predictors retain 66,860 valid raw printed-f labels from 6,686 molecules, with ten states and no exclusions. Recomputed pooled R² is 1−SSE/SST over flattened state labels, with SST 158.4062371581119. Each state's count is 6,686; per-state and four brightness-bin SSEs sum to pooled SSE. Fixed TRAIN thresholds 0.0549 and 0.2412 retain 6,650 and 639 true-tail labels. Reported RMSE, R², contrasts, eligibility and selected minimum SSE are internally consistent.

## Scientific result

| Recipe | Legacy validation pooled raw-f R² | Change versus its affine control |
|---|---:|---:|
| Historical affine incumbent | 0.418119244 | — |
| In-sample affine | 0.402024351 | — |
| In-sample fixed hinge | 0.398331068 | −0.003693283 |
| Held-out-source affine | 0.416374515 | — |
| Held-out-source fixed hinge | 0.411775449 | −0.004599065 |

Neither hinge meets the frozen ≥0.003 improvement requirement against both its own affine control and the incumbent. The held-out-source hinge exceeds the in-sample hinge by 0.013444381, but this source-procedure contrast is not a benefit of added scalar flexibility. Its source also differs in training size and prediction quality; it does not isolate membership alone.

The held-out-source hinge improves only S5 versus its affine control; nine states worsen, with the largest SSE regression in S7. Both hinges worsen true q90 and q99 RMSE against their own affine controls. The held-out hinge reduces false-bright count from 86 to 70 while increasing that bin's SSE from 9.71686 to 10.10856. Relative to the incumbent it has lower MAE and better true-tail RMSE but worse pooled raw-f R². Retaining the incumbent reflects the primary metric, not uniform dominance on every metric.

The fixed four-dimensional basis improves fitting SSE but loses transfer accuracy. Both fitted maps have nonnegative segment slopes; no calibration or validation output was clamped, and all validation inputs lie within the corresponding fitting range. These facts do not identify the cause of the regression. Paired descriptive bootstrap intervals for either hinge against its affine control include zero. The result supports stopping this fixed proposal under its declared allocation rule, not a universal claim that nonlinear readouts cannot help.

## Scope and limitations

This is a repeat exposure to the historical outer validation set. Its best affine incumbent was fitted on that same validation set; historical OOF calibration evidence must be reported separately. No old or new test scoring occurred in Round04. There is no new independent external confirmation, and the proposed versioned molecular-group partition has not been materialized or verified. Existing checkpoint training exposure means it cannot serve as a fresh model on future repartitioned data.

Each deployment contains the same full eta0 backbone and one scalar map. The temporary 80%-TRAIN source is absent at inference; there is no model/checkpoint averaging and no quantum-chemical input requirement. E and A remain unchanged, so mapped f need not equal the original energy/trace(A) relation. The map is empirical calibration, not an identified response operator.

Science's ROUND04_REPORT.md is accurate and suitable for publication. Reproduction commands document the completed, immutable run; completion markers intentionally prevent repeat evaluation. Preserve all old snapshots and keep private checkpoints, learned coefficient tensors, datasets, memberships and prediction caches off Git. Publish lightweight records only after the verified server-to-D archive.

The associated JSON/manifest binds exact source, execution, aggregate report and reviewer-script bytes. The root owns the final research/publication decision; this independent review makes no authorization for a new experiment.
