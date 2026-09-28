# Independent scratch native-arm review

Reviewed at: 2026-09-29T07:08:59.947514+08:00

**Integrity PASS for this completed individual arm. Paired scratch conclusions remain pending.** No MTO outcome, other active study metric, test label or new model inference was accessed.

| Checkpoint | Full-TRAIN raw-f R² | Validation raw-f R² | Validation SSE |
|---|---:|---:|---:|
| Initial | 0.002234759 | 0.002241607 | 158.051153 |
| Selected epoch 21 | 0.584645710 | 0.343781976 | 103.949028 |
| Final epoch 100 | 0.914207968 | 0.227522441 | 122.365263 |

Both raw-f and joint-objective selection choose epoch 21; their saved f prediction arrays are exactly equal. Thus the weak selected native score cannot be explained solely by using joint-selected rather than available raw-f-selected checkpoints in this run. The full 100-epoch schedule completed, with 188,100 steps, all 100 archived batch orders and the prescribed learning rates.

The final model fits training much better but validation SSE rises 18.416235 from the selected checkpoint. Validation f MAE improves slightly (0.0187042 to 0.0184089), and energy MAE improves (0.120936 to 0.108704 eV), so these metrics do not substitute for the pooled raw-f R² objective. This is evidence of poor generalization during this optimization trajectory, not proof that the shared backbone cannot represent the targets, that all direct readouts fail, or that a particular tail mechanism caused the loss. No additional tail/geometry inference is made from the aggregate result.

Independent checks verified 58 source and 10 code hashes, launch/review/preflight bindings, terminal and selected/versioned checkpoint-array hashes, all 100 order hashes, strict best selection including epoch 0, fixed full-TRAIN counts/arithmetic, exact validation IDs/indices/raw FP64 labels, finite nonnegative native-f predictions and unchanged admitted GPU5 ECC. All four saved validation checkpoint arrays were independently recomputed; values agree exactly with the audit. Small differences against fixed-checkpoint metrics are the existing FP32 replays performed during authorized training completion; this audit performs no replay. Primary scores above use saved arrays. Full-TRAIN metrics remain producer aggregates because no full train prediction arrays were saved for elementwise independent verification.

Keep the completed arm unchanged and do not extend it or promote it to test. Reuse this integrity proof when both scratch arms complete; only then evaluate the prescribed pair, fixed scratch mean and permitted supplemental references. The separately authorized descriptor implementation/preflight does not authorize a full fit or validation run.

Evidence: NATIVE_INDEPENDENT_VERIFICATION.json; NATIVE_COMPLETION_AUDIT.json; immutable native FIT_COMPLETE and FIXED_CHECKPOINT_METRICS receipts. The companion provenance pins these artifacts.
