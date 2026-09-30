# Round05 independent preparation review

**PASS for preparation. Production fitting remains blocked.** The frozen manifest is `66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1` (82 files). This review does not authorize a fit or repeat any completed preparation stage.

## Scientific contract

The four arms start from identical fresh original MTO tensors and use new TRAIN statistics. They compare control, the existing right-operand F, raw-M decorrelation, and both for 60 fixed epochs. The original LE+Ls objective, LR .001 AMSGrad, batch 64, clip 5 and order seed 11 are matched. F is identity-initialized; inactive F tensors in controls do not create an active capacity-matched control. The weak penalty acts on untransformed learned states, not dipoles. Electronic phase covariance and physical state orthogonality are not claimed.

The reviewed v2 partition retains all 133,727 rows: 120,355 TRAIN, 6,686 validation and 6,686 TEST. Independent reconstruction found no overlap under the documented conservative identity rules. The new TEST includes 5,989 old TRAIN, 361 old validation and 336 old TEST molecules, so it is a new partition of historically exposed data. Fresh initialization and TRAIN-only normalization are mandatory; old-split scores are not directly comparable.

The evaluator retains all 66,860 validation labels, uses pooled raw-f SSE for earliest-best checkpoint selection with epoch 0 eligible, and reports per-state errors, TRAIN-defined bright tails, false-bright bins and energy errors. No TEST reader or scoring path is included. Selected-best and fixed-epoch comparisons must remain distinct. No averaging across models or checkpoints is used.

## Evidence checked

- Rehashed all 82 current frozen files and checked 161 source entries in completed receipts. Independently regenerated all 60 training-order hashes from private TRAIN index metadata only.
- Reviewed the selected-row decoder, masks, index/hash checks and completed TRAIN statistics audit. Synthetic reader checks exercised unauthorized partition/row/member rejection. The independent statistics review checked source, access ledger and formula tests; it did not repeat the complete raw numerical reduction.
- Synthetic authorization checks exercised 20 valid/invalid binding paths, including rejection before model/data imports or run artifacts. The production launcher and worker require a distinct exact authorization, matching review, all source hashes and a remotely verified D-first publication receipt.
- Reviewed CPU geometry, symmetry, masking and one-checkpoint load/forward evidence; the loader uses embedded model/config/statistics and requires no external training data or QC labels.
- Reviewed the completed GPU retry, owned process identity, admission and terminal receipts: exactly 12 disposable updates on 128 TRAIN molecules, no validation/TEST values. Model replay differed by at most 1.1921e-7, optimizer state by 2.9802e-8, loss by zero; order/RNG and shared-M identity were exact. No independent inference or training was repeated during this review.
- Reviewed recovery and durability: committed `last.pt` holds optimizer/RNG/order/history and selected-artifact hashes; incomplete epochs restart from it. Immutable selected snapshots and predictions are eligible only when referenced by committed state. Completed arms refuse relaunch. The geometry export contains one complete predictor checkpoint.

## Preserved corrections and limits

The first CPU fixture changed `mto.n_ref` dtype through whole-model double-to-float conversion; its failure and source remain preserved. Native-dtype reconstruction repaired the fixture without relaxing the buffer check.

The first GPU attempt failed full-forward bitwise identity before any update. A separate zero-update diagnostic found repeat variability within identical models while shared-M coupling was exact. Root explicitly amended full-GPU identity/export acceptance to atol 2e-6, rtol 1e-5 after this evidence; this was not an originally unchanged criterion. No particular CUDA kernel was isolated. Resume criteria and the 12-update budget were unchanged.

After the successful retry, only parent-directory fsync blocks were added to two I/O helpers. The exact executed source, mapping and independent AST review are preserved. All objective, update, selection and RNG nodes are unchanged. The freezer accepts only this explicit historical-source mapping; other mismatches fail. No I/O behavior test or new numerical preflight is claimed for that correction.

The final metadata checker initially encountered Python 3.10 versus 3.12 AST serialization differences for an empty `type_params` field. It normalized that representation and passed without modifying frozen scientific files or reading targets.

The 128-row fixture does not establish full-corpus throughput, worst-case memory, generalization or an accuracy improvement. Epoch TRAIN diagnostics aggregate pre-update minibatches along the trajectory; validation uses a completed checkpoint. CUDA training is not promised bitwise reproducible. All technical model states remain private and are excluded from production initialization and publication.

## Remaining gates

Root preparation acceptance and download-first archival publication must precede a separate scope-specific execution authorization. GPU health and occupancy must be rechecked immediately at launch. No production run artifacts or execution authorization were present at the final binding check. The target R² 0.60 has not been achieved by this preparation.
