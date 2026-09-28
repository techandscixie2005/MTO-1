# Monitoring scope amendment

Updated 2026-09-29. This is an administrative scope correction only. It preserves `first_health_check_at` / `last_full_check_at` at `2026-09-28T22:29:29+08:00` and `next_check_due_at` at `2026-09-29T02:29:29+08:00`. No routine training metrics or logs were inspected for this change.

The existing heartbeat `qm9s-e-a` remains ACTIVE with `FREQ=HOURLY;INTERVAL=4`, targeting the same thread. Its prompt now covers dynamic discovery from the coordinator, both loss and architecture queues, all four architecture arms (`retained_residual`, `original`, `direct_f`, `independent_trace`), explicit G2 administrative-stop/partial-campaign provenance, and G4 natural completion at epoch 223.

Last-known handles recorded in `monitoring_state.json` are historical references only: loss queue 785736/control worker 785960 on GPU2; architecture queue 778242/retained-residual worker 778585 on guarded GPU5; original-control worker 790200 on GPU6. Current identity and health must be rediscovered at a scheduled check. G2 PID 566514 was intentionally `ADMIN_STOPPED` at epoch 215/cursor 114496; old queue 762340 exited without workers.

Resource policy: exclude GPUs 3 and 7; allow guarded GPU5 with existing admission safeguards; preserve unrelated GPU0 work; use other devices only when healthy, free, and admitted. The amendment does not claim any listed PID is currently healthy and does not change experiment protocols or coordinator result documents.
