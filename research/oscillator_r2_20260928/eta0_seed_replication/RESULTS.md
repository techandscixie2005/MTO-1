# Eta0 seed23/37 replication

Validation raw native-f results from fixed checkpoint rules.

| Predictor | R² | ΔR² vs seed11 | RMSE |
| --- | ---: | ---: | ---: |
| seed11_legacy | 0.405294 | +0.000000 | 0.037537 |
| seed23_legacy | 0.368709 | -0.036585 | 0.038674 |
| seed37_legacy | 0.394455 | -0.010839 | 0.037877 |
| primary_equal_seeds_11_23_37 | 0.452955 | +0.047661 | 0.036001 |
| secondary_mixed_selection_equal_three | 0.452955 | +0.047661 | 0.036001 |
| fixed_equal_three | 0.449421 | +0.044127 | 0.036117 |

Primary averages three legacy-selected seed checkpoints. The secondary average mixes old legacy selection with new raw-f selection.
Full paired molecule/connectivity bootstrap, state and tail diagnostics are in RESULTS.json.
No test access or ensemble-weight fitting.
