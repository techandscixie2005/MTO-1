# Frozen eta0 residual-head validation result

One reviewed 20-epoch, seed-11 head-only run; source eta0 features, energy and tensor base stayed frozen.

| Predictor | Epoch | Raw native-f R² | ΔR² vs eta0 | f RMSE | E MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| eta0 control | 33 source | 0.405294 | 0 | 0.037537 | 0.091388 |
| frozen head initial | 0 | 0.405294 | +0.000000 | 0.037537 | 0.091388 |
| frozen head selected | 2 | 0.406790 | +0.001496 | 0.037489 | 0.091388 |
| frozen head final20 | 20 | 0.404299 | -0.000995 | 0.037568 | 0.091388 |

Only the validation-selected epoch is a candidate. The prior jointly trained residual selected epoch0 and ended epoch20 at R² 0.304938.
Molecule and connectivity-group bootstrap intervals and per-state/tail metrics are in FROZEN_RESIDUAL_RESULTS.json.
No test predictions were opened.
