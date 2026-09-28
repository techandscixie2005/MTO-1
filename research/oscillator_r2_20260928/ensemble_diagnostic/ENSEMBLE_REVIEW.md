# Independent ensemble diagnostic review

**PASS for one frozen exploratory historical-test comparison.** Reviewer did not access test predictions.

Verified exact molecule IDs, row indices, raw f truth, masks, prediction shape and all three input hashes. Independently reproduced pooled float64 raw-f R2: eta0 0.4052941183410983; equal-three 0.4494214632744171; gain0.0441273449333 and SSE reduction7.4200%. The script accesses validation predictions only. Fivefold simplex weights fit four molecule folds and predict the fifth correctly.

Protocol modification time precedes script and metrics, consistent with candidate specification before scoring; timestamps are not immutable proof. The base checkpoints already used validation selection. Equal-three is now selected after comparison with the reported pairs/OOF method, so validation improvement is exploratory and OOF does not remove all selection optimism.

Freeze existing eta0/eta01/eta1 checkpoints and weights1/3 each. Average their native f values directly. Do not substitute c*mean(E)*tr(mean(A)), which introduces different cross terms. Compare only this ensemble with eta0 on the historical test after writing the selection record; use molecule-paired uncertainty and no test tuning. Report this as an ensemble improvement separately from the single-model benchmark, which remains eta0. The historical test has already been exposed and cannot establish fresh confirmation.

Residual correlations0.842-0.871 and averaging gains show useful error cancellation; they do not by themselves prove variance is the sole bottleneck. Training queue/configuration remains unchanged.
