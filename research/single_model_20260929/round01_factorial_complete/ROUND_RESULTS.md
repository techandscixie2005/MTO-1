# Four-arm single-model pilot results

All results use the fixed validation set. Every row is one model and one selected checkpoint. No prediction averaging or test inference occurred. Intervals are descriptive after checkpoint and validation reuse.

| Arm | Selected epoch | Native raw-f R2 | Delta vs control | Delta vs epoch0 | Fixed calibration R2 |
|---|---:|---:|---:|---:|---:|
| control | 0 | 0.405294116 | +0.000000000 | +0.000000002 | 0.418119236 |
| adapter | 0 | 0.405294117 | +0.000000001 | +0.000000001 | 0.418119236 |
| decorrelation | 0 | 0.405294117 | +0.000000001 | +0.000000011 | 0.418119238 |
| both | 0 | 0.405294115 | -0.000000001 | -0.000000014 | 0.418119236 |

## Matched final epoch (epoch20)

| Arm | Native raw-f R2 | Fixed calibration R2 |
|---|---:|---:|
| control | 0.366655307 | 0.394554483 |
| adapter | 0.366210889 | 0.394237328 |
| decorrelation | 0.366738959 | 0.394629619 |
| both | 0.366296425 | 0.394313638 |

Native same-epoch factorial contrast R2(both)-R2(adapter)-R2(decorrelation)+R2(control): +0.000001884.
Selected-best contrast is -0.000000003 and is only a descriptive pipeline comparison; separately selected epochs do not establish mechanistic synergy.

All sample-order hashes match. Per-state, bright-tail SSE/RMSE/MAE, energy, checkpoint SHA256 and paired molecule bootstrap details are in ROUND_RESULTS.json.

The fixed historical calibration is secondary, never refit, and did not select checkpoints. Its validation estimate uses the previously refitted map and is not a fresh holdout result.

## Next decision

Prespecified confirmation priorities: .
A control-only gain can justify paired independent-seed confirmation of the gentler schedule. An adapter win would still need a simpler active capacity control before attributing benefit to its nonlinearity. Root reviews state/tail tradeoffs and chooses subsequent work.
