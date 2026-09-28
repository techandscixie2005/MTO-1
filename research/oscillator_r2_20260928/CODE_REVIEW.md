# Independent code review

**PASS for the frozen queued A/B/C screen.** Reviewer did not launch training.

Reviewed: train.py, preflight.py, pilot_queue.py, summarize.py, pilot_objective.py, pilot_config.json. Exact approved hashes and the preflight hash are in CODE_REVIEW.json. All35 source hashes were independently compared with current files.

Verified raw-f target/indexing, valid-label masks, c*E*tr(A), retained predicted-E gradients, matched loss scale, float64 pooled metrics, epoch-zero eligibility and reproduction gate, identical source weights/fresh optimizers/sample order, resume state/history consistency and selection aliases. CPU checks reproduce saved validation R2=0.4052941183410983; optimizer next-step resume difference is exactly0. Independent synthetic masked-NaN gradients are finite and the weighted/direct-f identity holds.

Fixed before approval: FP32 trace accumulation in evaluation, stale preflight/data checks, aliased history dictionaries, incomplete selection recovery, missing order evidence, queue receipt directory race, obsolete summary config key, and queue.py shadowing Python standard-library queue. Launcher is now pilot_queue.py.

Resource review: GPU2 only after old G2 naturally completes/exits, clean health/idle-memory checks, shared GPU advisory lock retained by worker, repeated conflict checks, no resets/preemption, and deference to old campaign evaluation. Queue refuses changed approved files or stale preflight. Frozen artifacts stay on server; archive code/config/reports only.

Limitations: GPU epoch-zero inference remains a mandatory runtime gate before updates. CUDA aggregation is not claimed bitwise deterministic. Validation selection/bootstrap is exploratory; no test evaluation occurs in this screen. Energy regression is reported as a tradeoff, not an automatic veto of oscillator-strength improvement.
