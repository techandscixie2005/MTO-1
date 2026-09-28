# Seed23 individual completion audit

PASS: 100 epochs and 188,100 updates. Reviewed source and 100 order hashes, selected checkpoint/array hashes, raw FP64 validation labels, and GPU1 ECC counters match. Seed37 and the three-seed family result remain pending. No test access or audit-time inference.

| Checkpoint | Train f R² | Val f R² | Val SSE | Val f MAE | Val E MAE |
|---|---:|---:|---:|---:|---:|
| initial (epoch 0) | -0.058630 | -0.060210 | 167.943901 | 0.019997 | 0.558215 |
| legacy (epoch 53) | 0.778586 | 0.368709 | 100.000448 | 0.017705 | 0.087962 |
| raw_f (epoch 53) | 0.778586 | 0.368709 | 100.000448 | 0.017705 | 0.087962 |
| final100 (epoch 100) | 0.884062 | 0.336711 | 105.069155 | 0.017417 | 0.084683 |

Legacy and raw-f selection both chose epoch53; their saved validation arrays have the same SHA-256. Train metrics are completion-time full-train aggregates; saved validation metrics are recomputed here. Saved-array versus completion-time replay differences are recorded within the declared FP32 tolerances.

The primary family comparison uses legacy-selected seeds after both complete. Raw-f-selected and mixed-selection outputs are separately labeled secondary analyses. No family comparison or promotion is made here.
