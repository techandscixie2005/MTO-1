# Matched oscillator-strength continuation screen

All results below use the frozen validation split and the original raw oscillator-strength label. No test split was opened.

| Arm | Best epoch | Native-f val R² | ΔR² vs control | Bootstrap 95% CI | Energy MAE ratio |
| --- | ---: | ---: | ---: | --- | ---: |
| control | 0 | 0.405294 | +0.000000 | [+0.000000, +0.000000] | 1.000 |
| weighted | 0 | 0.405294 | +0.000000 | [-0.000000, +0.000000] | 1.000 |
| direct_f_matched | 0 | 0.405294 | +0.000000 | [-0.000000, +0.000000] | 1.000 |

## Interpretation
The raw metrics, per-state scores, bright-state SSE, and checkpoint-selection comparisons are in pilot_summary.json. The research coordinator should decide whether any arm merits replication; this script makes no test-set or promotion decision.
