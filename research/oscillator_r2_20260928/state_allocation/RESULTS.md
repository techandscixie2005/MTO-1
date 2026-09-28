# Validation state-allocation diagnostic

Primary metrics retain the original state ordering and printed raw oscillator strengths. Equal-three is an ensemble; the eta models are single models. No training or test predictions were used.

| Model | Raw native-f validation R² | MAE | Energy-order inversions |
| --- | ---: | ---: | ---: |
| mto_eta0 | 0.405294 | 0.017786 | 83 |
| mto_eta01 | 0.377127 | 0.017724 | 99 |
| mto_eta1 | 0.387367 | 0.018188 | 110 |
| equal_three | 0.449421 | 0.017345 | 17 |

## Separate disjoint-pair diagnostics

Ratio = pair-summed SSE / separate-state SSE. Oracle fraction is a truth-informed local swap bound, not attainable accuracy or R². Offsets must not be pooled.

| Offset | Gap | Model | Pairs | SSE | Sum ratio | Error correlation | Oracle swap fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| offset0 | low | mto_eta0 | 3352 | 10.75967 | 0.7726 | -0.2292 | 0.2080 |
| offset0 | low | equal_three | 3352 | 9.92103 | 0.7164 | -0.2867 | 0.2033 |
| offset0 | high | mto_eta0 | 3986 | 9.42462 | 0.9059 | -0.1011 | 0.0687 |
| offset0 | high | equal_three | 3986 | 7.29100 | 0.8983 | -0.1073 | 0.0644 |
| offset1 | low | mto_eta0 | 2673 | 7.34726 | 0.6795 | -0.3230 | 0.3027 |
| offset1 | low | equal_three | 2673 | 6.48557 | 0.6664 | -0.3363 | 0.2529 |
| offset1 | high | mto_eta0 | 2032 | 10.33088 | 0.9025 | -0.1027 | 0.0781 |
| offset1 | high | equal_three | 2032 | 10.11396 | 0.8999 | -0.1144 | 0.0517 |

## Connectivity-group uncertainty

| Offset | Gap | Ensemble separate-SSE reduction, 95% interval | Change in sum ratio, 95% interval |
| --- | --- | --- | --- |
| offset0 | low | [-0.0031, +0.1625] | [-0.1192, +0.0073] |
| offset0 | high | [+0.1226, +0.3504] | [-0.0659, +0.0475] |
| offset1 | low | [+0.0547, +0.1715] | [-0.0576, +0.0284] |
| offset1 | high | [-0.1382, +0.2475] | [-0.0617, +0.0335] |

Gap cuts: 0.04250 and 0.53737 eV. Pair-strength q50/q90: 0.024100/0.113600. Full state-pair × gap × brightness strata are in RESULTS.json. Group bootstrap uses2000 replicates over6671 connectivity groups.

These are conditional exploratory diagnostics. Small-gap cancellation can reflect intensity allocation, state composition, rounding or prediction bias; it does not establish physical mixing or justify relabelling. Root should review strata alongside current objective/readout results before choosing another architecture.
