# Matched MTO architecture screen

All figures use the frozen validation split and original raw oscillator-strength labels. No historical test predictions were opened.

| Arm | Best epoch | Native-f validation R² | ΔR² versus original | Bootstrap 95% interval | Energy MAE ratio |
| --- | ---: | ---: | ---: | --- | ---: |
| original | 0 | 0.405294 | +0.000000 | [+0.000000, +0.000000] | 1.000 |
| retained_residual | 0 | 0.405294 | +0.000000 | [-0.000000, +0.000000] | 1.000 |
| direct_f | 1 | 0.369250 | -0.036044 | [-0.109679, +0.007562] | 0.983 |
| independent_trace | 1 | 0.372023 | -0.033272 | [-0.099414, +0.008236] | 0.985 |

Full per-state, bright-transition, energy, initialization and objective-selection metrics are in ARCHITECTURE_RESULTS.json. The research coordinator decides whether any arm merits extension or replication.
