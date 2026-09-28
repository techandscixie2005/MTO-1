# Administrative retirement of G2

G2 was intentionally stopped on 2026-09-28 23:41:50 CST. This is an incomplete run, not a natural FIT_COMPLETE result.

**Decision.** The best validation objective remained at epoch 41 through the active epoch 215. The third learning-rate reduction at epoch 194 moves the earliest natural stopping point to epoch 244, so continuing would occupy GPU2 for a low-information interval. G2 cannot yield deployable native-f predictions because E was unsupervised.

The original waiting loss-pilot queue PID 762340 exited before G2 received SIGTERM. No pilot worker was active. The exact G2 process PID 566514 with start tick 1242573409, working directory /home/inspur/MTO-1/experiments/qm9s_chan64_20260928, and GPU2 receipt was rechecked immediately before signaling. The existing trainer handled SIGTERM and exited at the verified `within_epoch` checkpoint boundary (actual status: `STOPPED_CHECKPOINTED`). The original supervisor may later classify the four-group campaign as blocked; that status is expected and must not be overwritten.

| State | Epoch | Cursor | Steps | Best epoch | LR reductions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 215 | 0 | 402534 | 41 | 3 |
| After | 215 | 114496 | 404323 | 41 | 3 |

The resumable last checkpoint, optimizer, scheduler, all RNG states, and unchanged best checkpoint remain on the server. Their SHA-256 digests and process identity are in G2_ADMIN_STOPPED.json. No FIT_COMPLETE marker was created, and no GPU reset occurred. The replacement loss-pilot queue may use GPU2 only after independent idle/health/lock checks.
