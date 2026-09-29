# Train-only mechanism diagnostics

Fixed256-molecule training subset; no selection uses these aggregates. Predictions and feature arrays were not written. Full quantiles and per-state drift appear in MECHANISM_RESULTS.json.

| Arm/checkpoint | Epoch | Mean raw overlap squared | Mean relative adapter change | Native-f drift RMSE | Energy drift RMSE |
|---|---:|---:|---:|---:|---:|
| control/best | 0 | 0.438463 | 0 | 0 | 0 |
| control/last | 20 | 0.421382 | 0 | 0.00939919 | 0.0424297 |
| adapter/best | 0 | 0.438463 | 0 | 0 | 0 |
| adapter/last | 20 | 0.421355 | 0.00754326 | 0.00940324 | 0.0424187 |
| decorrelation/best | 0 | 0.438463 | 0 | 0 | 0 |
| decorrelation/last | 20 | 0.412794 | 0 | 0.00941371 | 0.0424657 |
| both/best | 0 | 0.438463 | 0 | 0 | 0 |
| both/last | 20 | 0.412703 | 0.00779328 | 0.00941752 | 0.0424543 |

Baseline mean raw overlap squared: 0.438463.
A changed latent overlap or adapter output demonstrates optimization effect, not physical wavefunction decorrelation or a causally identified error source.
