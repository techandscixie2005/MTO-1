# Submission receipt — 2026-09-21

Actual remote workspace: `/data/run01/sczc698/xxy/MTO`.
Independent experiment: `/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921`.
Remote code commit: `f5983a0be7673a91a51e104e51dd6619d5df3d17`, branch `codex/seed11-scales`. This is a separate local Git repository on the cluster; it was not pushed to GitHub.

| Step | Job ID | Required successful predecessor |
|---|---|---|
| Data preparation | 1299664 | none |
| 1k A/B array | 1299665_[0-1] | 1299664 |
| 1k test/report | 1299666 | all tasks of 1299665 |
| 10k A/B array | 1299667_[0-1] | 1299666 |
| 10k test/report | 1299668 | all tasks of 1299667 |
| Full A/B array | 1299669_[0-1] | 1299668 |
| Full test/final report | 1299670 | all tasks of 1299669 |

All dependencies were verified in Slurm as `afterok`. Array task 0 is original DetaNet; task 1 is planned MTO. Each task requests one GPU; maximum array concurrency is two. The script was invoked a second time and reused all seven existing job IDs without submitting duplicates.

Status at handoff: preparation PENDING (Priority); all remaining jobs PENDING (Dependency). **0/6 formal fits complete. No short forward/backward checks have executed yet, and no test results or stage-completion markers exist.** The short checks run automatically before each fresh fit on the allocated GPU, after real data preparation succeeds.

Verified before submission: source/split locations and existing scale counts; old experiment Git status clean; two unrelated jobs untouched; Python compilation; Bash syntax; accepted Slurm resources and dependencies; idempotent submission. Full data conversion and runtime finite-value checks remain queued and must not be described as passed.

See `README.md` for exact configuration, logs, outputs, status commands, and checkpoint recovery. Existing unrelated jobs `1297132` and `1289645` were preserved.
