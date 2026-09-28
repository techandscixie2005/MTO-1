# Scratch native completion audit

PASS: 100 epochs, 188,100 updates; reviewed source/data and 100 order hashes, selected checkpoint/array hashes, FP64 validation labels, and GPU5 ECC counters match. MTO arm and paired result remain pending. No test or new inference.

| Checkpoint | Train f R² | Val f R² | Val SSE | Val f MAE | Val E MAE |
|---|---:|---:|---:|---:|---:|
| initial (epoch 0) | 0.002235 | 0.002242 | 158.051153 | 0.024611 | 0.559283 |
| raw_f (epoch 21) | 0.584646 | 0.343782 | 103.949028 | 0.018704 | 0.120936 |
| joint (epoch 21) | 0.584646 | 0.343782 | 103.949028 | 0.018704 | 0.120936 |
| final100 (epoch 100) | 0.914208 | 0.227522 | 122.365263 | 0.018409 | 0.108704 |

Raw-f and joint selection both chose epoch 21. Separate checkpoint and array aliases are retained; completion-time replay aggregates show the reported FP32 variation. The two saved selected-validation arrays have identical f metrics.
Train metrics are terminal full-train aggregates; validation metrics were independently recomputed from saved predictions. Saved arrays and a later fixed-checkpoint replay differ by at most 9.4e-7 in f SSE and 3.7e-5 in energy SSE; the audit uses explicit 2e-6/1e-4 absolute replay tolerances. Those replays were produced during the authorized training completion; this audit ran no model inference.
