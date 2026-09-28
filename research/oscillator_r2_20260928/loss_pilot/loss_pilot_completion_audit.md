# Loss-pilot completion integrity audit

CPU-only audit of the completed validation-only loss screen. No test set or new inference was used. Saved selected-epoch predictions were read remotely and are not part of the archive.

| Arm | Raw native-f val R² | SSE | MAE | RMSE | Best epoch |
|---|---:|---:|---:|---:|---:|
| control | 0.405294107664 | 94.205122620762 | 0.017785608349 | 0.037536525759 | 0 |
| weighted | 0.405294108896 | 94.205122425525 | 0.017785608345 | 0.037536525721 | 0 |
| direct_f_matched | 0.405294128034 | 94.205119393994 | 0.017785608331 | 0.037536525117 | 0 |

All arms have 21 history rows (baseline epoch 0 plus trained epochs 1–20). Epoch 0 has no training-order digest (`null`) by design; each trained epoch has a 64-character digest, and the three arms share the same digest at each epoch. Source hashes in each run manifest and terminal receipt match. All three arms ended with `FIT_COMPLETE`; no `FAILED.json` or `INVALID.json` marker exists. Molecule IDs, indices, and raw truths align across arms. The selected raw-f metrics above recompute from saved validation predictions. These results show a numerical tie and do not justify promotion, test evaluation, or extension.
