# Eta0 seed replication: completed validation audit

Both fresh eta0 seed runs completed 100 epochs and 188,100 updates under reviewed source and order hashes. Both selected checkpoints chose the minimum legacy validation objective. The secondary raw-f rule happened to select the same epoch and identical saved predictions in each run. Validation truth uses raw FP64 f for all 6,686 molecules and ten states. No test rows or audit-time model inference were accessed.

| Predictor | Raw native-f R² | SSE | Selection |
|---|---:|---:|---|
| seed11 eta0 | 0.405294 | 94.205121 | archived legacy |
| seed23 eta0 | 0.368709 | 100.000448 | legacy epoch 53 |
| seed37 eta0 | 0.394455 | 95.922052 | legacy epoch 44 |
| primary equal seed11/23/37 | 0.452955 | 86.655328 | three legacy-selected |
| fixed equal eta0/eta01/eta1 | 0.449421 | 87.215074 | prior fixed three |

Primary minus fixed-three ΔR² = +0.003534. Paired molecule 95% interval [-0.039888130042539896, 0.005215224367479631, 0.029298182981039957]; connectivity-group interval [-0.0374797861958643, 0.0050127598286046175, 0.02968244380916484]. Both include zero. This is exploratory evidence under reused validation and three-seed uncertainty.

The reviewed postrun geometry seed report preserves fixed train-derived q2 strata and q3 diagnostics. The separately approved fixed-seven secondary saved-array result is R² 0.486691, below frozen F5 0.487700; it is not a new test result. The original fixed7 sidecar failed on an omitted all-valid mask field in fresh seed NPZ files; an independently reviewed isolated adapter verified the authoritative mask, exact IDs/FP64 truth, hashes and original selection gates before the one corrected call. Original failure and source remain preserved.

The old fixed-three historical test R² 0.485102 does not transfer to either new ensemble. No new test evaluation, weight search, subset search or further training was performed.
