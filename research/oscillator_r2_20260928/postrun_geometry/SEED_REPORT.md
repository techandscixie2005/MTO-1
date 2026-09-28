# Post-run validation geometry comparison

Mode: completed seed family. All benchmark metrics retain the full validation set (6,686 molecules, 66,860 raw-f labels).
No fitting, model inference, checkpoint selection, sample removal or test access occurred.

## Full validation metrics

| Predictor and selection | SSE | R² | ΔR² vs eta0 | ΔR² vs fixed equal3 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| eta0_legacy_joint_selected | 94.205120929 | 0.405294 | +0.000000 | -0.044127 | 0.0177856 | 0.0375365 |
| fixed_equal3 | 87.215074263 | 0.449421 | +0.044127 | +0.000000 | 0.0173450 | 0.0361171 |
| seed23_legacy_joint_selected | 100.000448422 | 0.368709 | -0.036585 | -0.080713 | 0.0177049 | 0.0386739 |
| seed23_secondary_raw_f_selected | 100.000448422 | 0.368709 | -0.036585 | -0.080713 | 0.0177049 | 0.0386739 |
| seed37_legacy_joint_selected | 95.922052026 | 0.394455 | -0.010839 | -0.054966 | 0.0174980 | 0.0378770 |
| seed37_secondary_raw_f_selected | 95.922052026 | 0.394455 | -0.010839 | -0.054966 | 0.0174980 | 0.0378770 |
| primary_equal_seed11_23_37 | 86.655328163 | 0.452955 | +0.047661 | +0.003534 | 0.0170211 | 0.0360010 |
| secondary_mixed_selection_equal3 | 86.655328163 | 0.452955 | +0.047661 | +0.003534 | 0.0170211 | 0.0360010 |

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
| seed23_legacy_joint_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| seed23_legacy_joint_selected | q2 <= 1e-05 | 1 | 10.03699160 | +7.13898874 | +6.44045071 |
| seed23_legacy_joint_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 2.01629777 | -0.47138554 | +0.35592038 |
| seed23_legacy_joint_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.95962824 | +0.91583472 | +0.67087463 |
| seed23_legacy_joint_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 12.52857688 | -0.41799179 | +0.93593878 |
| seed23_legacy_joint_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 32.25961106 | -1.42403347 | +1.41474864 |
| seed23_legacy_joint_selected | q2 > 0.450098133123 | 3376 | 39.19934286 | +0.05391484 | +2.96744101 |
| seed23_secondary_raw_f_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| seed23_secondary_raw_f_selected | q2 <= 1e-05 | 1 | 10.03699160 | +7.13898874 | +6.44045071 |
| seed23_secondary_raw_f_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 2.01629777 | -0.47138554 | +0.35592038 |
| seed23_secondary_raw_f_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.95962824 | +0.91583472 | +0.67087463 |
| seed23_secondary_raw_f_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 12.52857688 | -0.41799179 | +0.93593878 |
| seed23_secondary_raw_f_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 32.25961106 | -1.42403347 | +1.41474864 |
| seed23_secondary_raw_f_selected | q2 > 0.450098133123 | 3376 | 39.19934286 | +0.05391484 | +2.96744101 |
| seed37_legacy_joint_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| seed37_legacy_joint_selected | q2 <= 1e-05 | 1 | 6.95818045 | +4.06017759 | +3.36163956 |
| seed37_legacy_joint_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 1.80581202 | -0.68187129 | +0.14543463 |
| seed37_legacy_joint_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 4.78991757 | +1.74612405 | +1.50116395 |
| seed37_legacy_joint_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 11.94411341 | -1.00245526 | +0.35147531 |
| seed37_legacy_joint_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 32.12452282 | -1.55912172 | +1.27966040 |
| seed37_legacy_joint_selected | q2 > 0.450098133123 | 3376 | 38.29950576 | -0.84592226 | +2.06760391 |
| seed37_secondary_raw_f_selected | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| seed37_secondary_raw_f_selected | q2 <= 1e-05 | 1 | 6.95818045 | +4.06017759 | +3.36163956 |
| seed37_secondary_raw_f_selected | 1e-05 < q2 <= 0.0594195634181 | 5 | 1.80581202 | -0.68187129 | +0.14543463 |
| seed37_secondary_raw_f_selected | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 4.78991757 | +1.74612405 | +1.50116395 |
| seed37_secondary_raw_f_selected | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 11.94411341 | -1.00245526 | +0.35147531 |
| seed37_secondary_raw_f_selected | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 32.12452282 | -1.55912172 | +1.27966040 |
| seed37_secondary_raw_f_selected | q2 > 0.450098133123 | 3376 | 38.29950576 | -0.84592226 | +2.06760391 |
| primary_equal_seed11_23_37 | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| primary_equal_seed11_23_37 | q2 <= 1e-05 | 1 | 6.25727574 | +3.35927287 | +2.66073484 |
| primary_equal_seed11_23_37 | 1e-05 < q2 <= 0.0594195634181 | 5 | 1.77721706 | -0.71046626 | +0.11683966 |
| primary_equal_seed11_23_37 | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.50794129 | +0.46414777 | +0.21918768 |
| primary_equal_seed11_23_37 | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 11.26194062 | -1.68462805 | -0.33069747 |
| primary_equal_seed11_23_37 | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 29.10512602 | -4.57851852 | -1.73973640 |
| primary_equal_seed11_23_37 | q2 > 0.450098133123 | 3376 | 34.74582744 | -4.39960059 | -1.48607441 |
| secondary_mixed_selection_equal3 | degenerate | 0 | 0.00000000 | +0.00000000 | +0.00000000 |
| secondary_mixed_selection_equal3 | q2 <= 1e-05 | 1 | 6.25727574 | +3.35927287 | +2.66073484 |
| secondary_mixed_selection_equal3 | 1e-05 < q2 <= 0.0594195634181 | 5 | 1.77721706 | -0.71046626 | +0.11683966 |
| secondary_mixed_selection_equal3 | 0.0594195634181 < q2 <= 0.0986709913548 | 59 | 3.50794129 | +0.46414777 | +0.21918768 |
| secondary_mixed_selection_equal3 | 0.0986709913548 < q2 <= 0.210339717291 | 597 | 11.26194062 | -1.68462805 | -0.33069747 |
| secondary_mixed_selection_equal3 | 0.210339717291 < q2 <= 0.450098133123 | 2648 | 29.10512602 | -4.57851852 | -1.73973640 |
| secondary_mixed_selection_equal3 | q2 > 0.450098133123 | 3376 | 34.74582744 | -4.39960059 | -1.48607441 |

## Interpretation limits

- All full benchmarks retain every molecule; case14562/six-case values are concentration diagnostics only.
- Geometry associations are descriptive and do not establish causality.
- New checkpoints were selected on the same reused validation set; deltas are exploratory and selection conditional.
- Round09 defined q2 bins only; q3 remains a ratio/quantile diagnostic without new bins.
- No test indices, labels, geometry, predictions or metrics were accessed.

Per-state SSE, signed errors, absolute delta SSE, case14562 and fixed six-case concentration are in the JSON.
