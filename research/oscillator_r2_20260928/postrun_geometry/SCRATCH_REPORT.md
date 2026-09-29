# Post-run validation geometry comparison

Mode: completed scratch family. All benchmark metrics retain the full validation set (6,686 molecules, 66,860 raw-f labels).
No fitting, model inference, checkpoint selection, sample removal or test access occurred.

## Full validation metrics

| Predictor and selection | SSE | R² | ΔR² vs eta0 | ΔR² vs fixed equal3 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| eta0_legacy_joint_selected | 94.205120929 | 0.405294 | +0.000000 | -0.044127 | 0.0177856 | 0.0375365 |
| fixed_equal3 | 87.215074263 | 0.449421 | +0.044127 | +0.000000 | 0.0173450 | 0.0361171 |
| scratch_native_raw_f_selected | 103.949027936 | 0.343782 | -0.061512 | -0.105639 | 0.0187042 | 0.0394300 |
| scratch_mto_raw_f_selected | 99.242624587 | 0.373493 | -0.031801 | -0.075929 | 0.0183802 | 0.0385271 |
| scratch_fixed_equal_mean_secondary | 94.547365302 | 0.403134 | -0.002161 | -0.046288 | 0.0181175 | 0.0376046 |

## Matched scratch contrast

Primary native minus MTO selected raw-f R²: -0.029711; ΔSSE +4.706403348.
The fixed 0.5 native + 0.5 MTO mean is a post-launch, pre-outcome secondary diagnostic; component and reference deltas are in JSON.

## Frozen geometry bins

q2 cuts: [1e-05, 0.05941956341810567, 0.09867099135475822, 0.21033971729099188, 0.45009813312285296]. These are from TRAIN only and right-closed; q3 has no Round09 bins.
Round09 TRAIN q3 quantiles (descriptive only): {'0.001': 1.3129937341794305e-11, '0.01': 1.0273606733704654e-09, '0.1': 0.03703241529916692, '0.5': 0.1950348664308274}.
Round09 eta0/equal3 q2 molecule counts and SSE reproduce exactly within 1e-10.

### q2 bins

| Predictor | Bin | Molecules | SSE | ΔSSE vs eta0 | ΔSSE vs equal3 |
|---|---|---:|---:|---:|---:|
| eta0_legacy_joint_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| eta0_legacy_joint_selected | q2 <= 1e-05 | 1 | 2.89800287 | +0.00000000 | -0.69853803 |
| eta0_legacy_joint_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 2.48768331 | +0.00000000 | +0.82730592 |
| eta0_legacy_joint_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.04379352 | +0.00000000 | -0.24496009 |
| eta0_legacy_joint_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 12.94656867 | +0.00000000 | +1.35393057 |
| eta0_legacy_joint_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 33.68364454 | +0.00000000 | +2.83878212 |
| eta0_legacy_joint_selected | q2 > 0.450098133123 | 3376 | 39.14542802 | +0.00000000 | +2.91352618 |
| fixed_equal3 | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| fixed_equal3 | q2 <= 1e-05 | 1 | 3.59654090 | +0.69853803 | +0.00000000 |
| fixed_equal3 | 1e-05 < q2 <= 0.0594195634181 | 5 | 1.66037739 | -0.82730592 | +0.00000000 |
| fixed_equal3 | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.28875361 | +0.24496009 | +0.00000000 |
| fixed_equal3 | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 11.59263809 | -1.35393057 | +0.00000000 |
| fixed_equal3 | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 30.84486242 | -2.83878212 | +0.00000000 |
| fixed_equal3 | q2 > 0.450098133123 | 3376 | 36.23190185 | -2.91352618 | +0.00000000 |
| scratch_native_raw_f_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| scratch_native_raw_f_selected | q2 <= 1e-05 | 1 | 3.99063732 | +1.09263446 | +0.39409643 |
| scratch_native_raw_f_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 2.57575133 | +0.08806802 | +0.91537394 |
| scratch_native_raw_f_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.79809697 | +0.75430345 | +0.50934336 |
| scratch_native_raw_f_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 13.61138138 | +0.66481271 | +2.01874328 |
| scratch_native_raw_f_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 36.85272102 | +3.16907648 | +6.00785860 |
| scratch_native_raw_f_selected | q2 > 0.450098133123 | 3376 | 43.12043991 | +3.97501189 | +6.88853806 |
| scratch_mto_raw_f_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| scratch_mto_raw_f_selected | q2 <= 1e-05 | 1 | 3.29597647 | +0.39797361 | -0.30056442 |
| scratch_mto_raw_f_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 3.28939639 | +0.80171308 | +1.62901900 |
| scratch_mto_raw_f_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.46966387 | +0.42587035 | +0.18091026 |
| scratch_mto_raw_f_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 12.95950634 | +0.01293767 | +1.36686825 |
| scratch_mto_raw_f_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 35.86344515 | +2.17980061 | +5.01858273 |
| scratch_mto_raw_f_selected | q2 > 0.450098133123 | 3376 | 40.36463635 | +1.21920833 | +4.13273451 |
| scratch_fixed_equal_mean_secondary | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| scratch_fixed_equal_mean_secondary | q2 <= 1e-05 | 1 | 3.63156327 | +0.73356040 | +0.03502237 |
| scratch_fixed_equal_mean_secondary | 1e-05 < q2 <= 0.0594195634181 | 5 | 2.72964445 | +0.24196113 | +1.06926705 |
| scratch_fixed_equal_mean_secondary | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.12088368 | +0.07709016 | -0.16786993 |
| scratch_fixed_equal_mean_secondary | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 12.40096986 | -0.54559881 | +0.80833176 |
| scratch_fixed_equal_mean_secondary | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 33.69042372 | +0.00677919 | +2.84556131 |
| scratch_fixed_equal_mean_secondary | q2 > 0.450098133123 | 3376 | 38.97388032 | -0.17154770 | +2.74197848 |

## Interpretation limits

- All full benchmarks retain every molecule; case14562/six-case values are concentration diagnostics only.
- Geometry associations are descriptive and do not establish causality.
- New checkpoints were selected on the same reused validation set; deltas are exploratory and selection conditional.
- Round09 defined q2 bins only; q3 remains a ratio/quantile diagnostic without new bins.
- No test indices, labels, geometry, predictions or metrics were accessed.

Per-state SSE, signed errors, absolute delta SSE, case14562 and fixed six-case concentration are in the JSON.
