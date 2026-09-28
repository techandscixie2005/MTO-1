# Architecture validation comparison addendum

Primary metric: pooled raw native oscillator-strength R². Each arm is one validation-selected single model. Comparisons use the original control from the same round.

| Arm | Epoch 0 R² | Selected epoch | Selected R² | Final epoch 20 R² | ΔR² vs original | Molecule 95% CI | Connectivity 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| original | 0.405294 | 0 | 0.405294 | 0.311987 | +0.000000 | [+0.00000, +0.00000] | [+0.00000, +0.00000] |
| retained_residual | 0.405294 | 0 | 0.405294 | 0.304938 | +0.000000 | [-0.00000, +0.00000] | [+0.00000, +0.00000] |
| direct_f | 0.334568 | 1 | 0.369250 | 0.305924 | -0.036044 | [-0.10968, +0.00756] | [-0.10728, +0.00558] |
| independent_trace | 0.335515 | 1 | 0.372023 | 0.314608 | -0.033272 | [-0.09941, +0.00824] | [-0.09716, +0.00670] |

## Epoch checkpoints

| Arm | Stage | Epoch | Raw-f R² | Raw-f MAE | Raw-f RMSE | E MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | epoch0 | 0 | 0.405294 | 0.017786 | 0.037537 | 0.091388 |
| original | selected_best | 0 | 0.405294 | 0.017786 | 0.037537 | 0.091388 |
| original | final20 | 20 | 0.311987 | 0.017601 | 0.040374 | 0.092399 |
| retained_residual | epoch0 | 0 | 0.405294 | 0.017786 | 0.037537 | 0.091388 |
| retained_residual | selected_best | 0 | 0.405294 | 0.017786 | 0.037537 | 0.091388 |
| retained_residual | final20 | 20 | 0.304938 | 0.017539 | 0.040580 | 0.092416 |
| direct_f | epoch0 | 0 | 0.334568 | 0.019092 | 0.039706 | 0.091388 |
| direct_f | selected_best | 1 | 0.369250 | 0.018189 | 0.038657 | 0.089863 |
| direct_f | final20 | 20 | 0.305924 | 0.017659 | 0.040551 | 0.093205 |
| independent_trace | epoch0 | 0 | 0.335515 | 0.019211 | 0.039678 | 0.091388 |
| independent_trace | selected_best | 1 | 0.372023 | 0.018168 | 0.038572 | 0.090063 |
| independent_trace | final20 | 20 | 0.314608 | 0.017587 | 0.040297 | 0.092909 |

Architecture raw-f error has 4.508882 times the weight of pilot direct_f_matched relative to the same energy term. Pilot control and weighted arms instead optimize trace losses. Compare architecture variants with architecture/original and objective-pilot variants with pilot/control. Architecture/original versus pilot/direct_f_matched is a loss-scale comparison, not an architecture gain.

The JSON includes raw-f MAE/RMSE, energy MAE at epoch 0/selected/final 20, every validation R² in the 21-epoch curve, per-state SSE contributions to pooled ΔR², fixed q90/q99 bright and complementary SSE, and A-derived-truth sensitivity.

Validation molecules: 6686; connectivity groups: 6671; repeated groups: 15; largest group: 2. Both paired bootstraps use 2000 replicates and keep all ten states together.

Bootstrap intervals condition on the selected checkpoints. They do not adjust for selection among 21 epochs or arms, and one seed does not measure training variability. The historical test is not read.
