# Architecture execution review

**PASS for the frozen four-arm screen and guarded dynamic queue.** The reviewer did not launch or monitor GPU training. EXECUTION_REVIEW.json records the44approved source hashes and preflight hash; all were independently compared with the current files and Python syntax parsed.

Verified common train-variance-scaled raw-f MSE plus unchanged energy loss; train-only8192molecule initialization; source/data/initial-checkpoint identity; seed/order/fixed20epoch budget; float64 pooled native-f validation; epoch-zero eligibility; original/residual runtime baseline gate; separate native-f/objective selection; consistent checkpoint/history recovery; and validation-only summary/paired resampling. Preflight optimizer next-step resume difference is exactly zero.

The executor fixed three findings before approval: GPU5 now matches the exact admitted ECC/remap baseline rather than accepting arbitrary corrected-error history; stopping the queue signals its own workers to checkpoint; old trainer.py process identity is checked correctly. Resource helper/admission report are now hash-bound. The queue uses only guardedGPU5 and clean releasedGPUs1/4/6, retains shared locks, preserves GPU2 for the loss pilot and avoids unrelated jobs. Fresh GPU5 microcheck and health snapshots surround execution; changed ECC invalidates the run.

Source model/config/protocol remain unchanged by this execution review. New scalar heads have additional initialization compute and omit tensor Gram information; residual keeps the original information and exact starting f. This remains a one-seed exploratory validation screen. Full GPU epoch-zero reproduction must succeed before any original/residual update; no historical-test evaluation is included.
