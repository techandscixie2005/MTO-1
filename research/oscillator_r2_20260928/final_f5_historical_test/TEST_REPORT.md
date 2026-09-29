# Frozen F5 historical-test comparison

The historical test has been reused in earlier campaigns; this is exploratory and not an untouched holdout.

| Predictor | Pooled raw-f SSE | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| eta0 | 103.193193599 | 0.455815109 | 0.03928641 | 0.01771599 |
| old3 | 97.639596726 | 0.485101765 | 0.03821464 | 0.01726514 |
| F5 | 93.178930153 | 0.508624899 | 0.03733152 | 0.01627730 |

Primary F5−old3 ΔR²: +0.023523134; relative SSE reduction +4.568502%.
Paired molecule bootstrap: [0.009246644613091175, 0.024806035535460912, 0.03913374393711249].
Paired connectivity bootstrap: [0.009644332708517065, 0.023908141508979894, 0.03900201119281694].
All states, TRAIN-derived tail and q2 slices, signed errors, and contribution/concentration diagnostics are in JSON.
Five model forwards are needed for deployment versus three for old3. Only two new selected chan64 forwards were run here.
No new training, tuning, clipping, model selection or test-driven change was made.
