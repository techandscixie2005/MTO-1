# Descriptor baseline completion audit

PASS: one reviewed 803-feature, 256-tree fit completed; fit froze before validation, source/input hashes and raw FP64 truth match. No test or audit-time inference.

| Predictor | Raw native-f R² | SSE | MAE |
|---|---:|---:|---:|
| D | 0.214527 | 124.423762 | 0.021454 |
| F5 | 0.487700 | 81.151578 | 0.016334 |
| equal_blend | 0.411543 | 93.215196 | 0.018462 |

Train D R² 0.486264; validation D R² 0.214527. Equal half-blend R² 0.411543 is below frozen F5 0.487700.
No model promotion, additional fit, validation search or test comparison follows from this result.
