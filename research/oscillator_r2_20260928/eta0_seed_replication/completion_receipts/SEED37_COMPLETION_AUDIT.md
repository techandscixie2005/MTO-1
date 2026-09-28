# Seed37 individual completion audit

PASS: 100 epochs and 188,100 updates. Reviewed source and 100 order hashes, selected checkpoint/array hashes, raw FP64 validation labels, and GPU1 ECC counters match. The three-seed family result is reported separately. No test access or audit-time inference.

| Checkpoint | Train f R² | Val f R² | Val SSE | Val f MAE | Val E MAE |
|---|---:|---:|---:|---:|---:|
| initial (epoch 0) | -0.047873 | -0.049029 | 166.172699 | 0.020243 | 0.559196 |
| legacy (epoch 44) | 0.725204 | 0.394455 | 95.922052 | 0.017498 | 0.090392 |
| raw_f (epoch 44) | 0.725204 | 0.394455 | 95.922052 | 0.017498 | 0.090392 |
| final100 (epoch 100) | 0.904943 | 0.352688 | 102.538225 | 0.017223 | 0.082417 |

Legacy and raw-f selection both chose epoch53; their saved validation arrays have the same SHA-256. Train metrics are completion-time full-train aggregates; saved validation metrics are recomputed here. Saved-array versus completion-time replay differences are recorded within the declared FP32 tolerances.

The primary family comparison uses legacy-selected seeds after both complete. Raw-f-selected and mixed-selection outputs are separately labeled secondary analyses. No family comparison or promotion is made here.
