# Completed retained-residual arm

The guarded GPU5 run finished its fixed 20 epochs with no ECC/remap change or recorded invalidation. Its selected raw-native-f checkpoint is epoch 0; later epochs did not improve the pooled validation SSE. This is one completed arm, so it is not a matched architecture comparison.

| Validation checkpoint | Epoch | Raw native-f R² | f MAE | f RMSE | E MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch0 | 0 | 0.405294130 | 0.017786 | 0.037537 | 0.091388 |
| selected_best | 0 | 0.405294130 | 0.017786 | 0.037537 | 0.091388 |
| final20 | 20 | 0.304937854 | 0.017539 | 0.040580 | 0.092416 |

Frozen eta0 reference R²: 0.405294118. Selected-minus-eta0: +0.000000011, consistent with numerical reproduction. Final-minus-epoch0: -0.100356276.

The selected prediction NPZ exactly matches frozen validation IDs, indices and raw labels; recomputed pooled metrics match history. Epoch20 predictions were not separately saved. Its metrics match both the committed last-checkpoint history and final status.

The original matched control and two other architecture arms are still needed before any promotion or architecture claim. The historical test was not opened.
