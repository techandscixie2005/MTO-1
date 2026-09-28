# Scheduled full monitoring check — 2026-09-29 06:29 slot

- Scheduled slot: 2026-09-29 06:29:29 +08.
- Observation began: 2026-09-29 06:29:39 +08.
- Observation ended: 2026-09-29 06:30:11 +08.
- Next scheduled check: 2026-09-29 10:29:29 +08 (existing four-hour cadence retained).
- Scope: the four receipt-pinned 100-epoch runs, their status/history/terminal markers, mapped GPUs 1/2/5/6, and current ECC/row-remap counters. No test evaluation, inference, or intervention.

## Process and training snapshot

All four `/proc` PID/start-tick identities, cwd and command line matched their reviewed launch receipts at 06:29:39. Every run was advancing and had no `FIT_COMPLETE`, `FAILED`, `INVALID` or `ADMIN_STOPPED` marker at the snapshot. Status and history were read at 06:30:11.

| Run | PID / GPU | Progress (status epoch / steps) | Best validation native-f R² (epoch) | Latest completed history epoch / R² | Latest SSE |
|---|---|---:|---:|---:|---:|
| eta0 seed 23 | 859155 / 1 | 83 / 154,400 | 0.36870 (53) | 81 / 0.34960 | 103.02672 |
| eta0 seed 37 | 859334 / 6 | 82 / 152,900 | 0.39445 (44) | 81 / 0.35600 | 102.01301 |
| scratch MTO | 879890 / 2 | 68 / 127,700 | 0.37350 (16) | 67 / 0.33406 | 105.48882 |
| scratch native | 880266 / 5 | 88 / 163,900 | 0.34378 (21) | 86 / 0.24144 | 120.16137 |

Best R² is derived from each run's recorded best raw-f validation SSE and the shared validation SST (158.406237). Latest values use the last complete `history.jsonl` row, not an incomplete in-progress epoch. All four current status losses and recorded gradient norms were finite; no OOM, NaN/Inf, traceback or failure marker appeared. The `train.log` files were empty, so progress evidence comes from advancing status/history records. Runs remain below their fixed 100-epoch limit; the scratch arms' recent validation scores are below their earlier best checkpoints, which is a validation-overfit warning, not a runtime failure.

## Device health

| GPU | Mapped run | Utilization / memory / temp | ECC corrected (volatile/aggregate) | ECC uncorrected | Remap pending / failure |
|---:|---|---|---:|---:|---|
| 1 | seed 23 | 37% / 4,123 of 81,920 MiB / 50 C | 0 / 0 | 0 / 0 | No / No |
| 2 | scratch MTO | 36% / 4,121 of 81,920 MiB / 48 C | 0 / 0 | 0 / 0 | No / No |
| 5 | scratch native | 41% / 4,113 of 81,920 MiB / 52 C | 1 / 1 | 0 / 0 | No / No |
| 6 | seed 37 | 42% / 4,121 of 81,920 MiB / 50 C | 0 / 0 | 0 / 0 | No / No |

GPU 5's one corrected DRAM ECC count matches its admission baseline; no new corrected or uncorrected errors, remap, pending repair, or remap failure was observed. GPU 5 remains guarded under its passed admission. GPUs 3 and 7 remain excluded; unrelated GPU 0 was not touched.

## Decision

No intervention is indicated. Continue the reviewed fixed runs to their 100-epoch limits; do not stop them based only on interim validation decline. No result is promoted and no test score was generated. Review completed checkpoints and saved validation outputs after each run emits an authoritative terminal receipt.
