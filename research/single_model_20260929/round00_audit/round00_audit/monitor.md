# Dedicated monitoring handoff

Updated: 2026-09-30 00:00 Asia/Shanghai. Active research scope `/home/inspur/MTO-1/research/single_model_20260929`.

## Machine audit

Server is an artifacts/source workspace without `.git`; `git` is absent from default server PATH. Local publication repo is `D:\MTO\publication\MTO-1`, with an unborn `main` working-tree HEAD, existing remote refs, and a preexisting untracked `PUBLICATION_TREE_IDS.json`. Preserve it. Local chat workspace has substantial preexisting modified/untracked historical records; do not stage broadly.

GPU 0 has unrelated SpecGPT training PID 712998 (supervisor 712791); never interfere. Preferred physical GPUs 1,2,4,6 have zero observed uncorrected ECC, no pending remapping, and were idle at admission audit. GPU 3 has volatile/aggregate uncorrected ECC 187/352 and pending remapping; GPU 7 has 2/2 and pending remapping. Both excluded. GPU 5 has one corrected ECC and is reserve. UUID mapping and policy are in `ops/registry.json`. Root filesystem has ~2.9 TiB free; `/data` ~11 TiB free.

Historical original 20-epoch continuation took 3057 seconds (~51 min) on GPU6. New four parallel pilots estimated 1–2 hours including validation and adapter overhead; revise after early epochs. No new fit jobs launched by monitoring agent.

## Persistent scheduling

Server user cron installed with one tagged block, preserving unrelated entries (none initially): `0 */4 * * * /usr/bin/python3 /home/inspur/MTO-1/research/single_model_20260929/ops/monitor.py --root /home/inspur/MTO-1/research/single_model_20260929 ...`. Runs at 00,04,08,12,16,20 server local Asia/Shanghai. Cron service active; user systemd had no timers and Linger=no, so cron chosen. Manual smoke record `monitoring/CHECK_20260929T155359Z.json` verified. Actual cron trigger verified at 2026-09-30 00:00:02+08:00, record `CHECK_20260929T160002Z.json`, 0 registered runs and 0 incidents; next scheduled run 04:00+08:00.

Existing app heartbeat `qm9s-e-a` independently verified ACTIVE, `FREQ=HOURLY;INTERVAL=4`, target current chat `01a0edd8-8c03-75d1-aeb0-953f11af6433`. Root updated this; no duplicate heartbeat was created. Read-only cron collects machine facts even if desktop is unavailable; heartbeat dispatches dedicated monitoring agent for diagnosis/recovery/archival and project decisions.

## Scripts and safety

Canonical server ops directory: `/home/inspur/MTO-1/research/single_model_20260929/ops`. Local authoring mirror `research_state/single_model_20260929/ops`.

- `monitor.py`: lock, explicit registry, `/proc` identity, progress/terminal files, checkpoint metadata only, NVIDIA health, timestamped snapshots/events. No process signals or restarts.
- `register_run.py`: scope-contained paths plus PID/start ticks/boot ID/UID/cwd/argv hash and GPU UUID. Require distinct attempt IDs for resumes. Registration command sent to science agent.
- `install_monitor.py`: one tagged cron block and schedule receipt, saves prior cron.
- `test_monitor_safety.py`: 11 read-only identity, containment, lifecycle, ECC checks pass.
- `package_records.py`: explicit text-file allowlist, size/path/credential screening, manifest and gzip archive.
- `download_records.py`: download FIRST to new `D:\MTO\archives\single_model_20260929\<round>`, verify SHA256 and exact member list, safe extraction, download receipt. Existing archives cannot be overwritten.
- `publish_records.py`: stage only downloaded verified records through separate Git index, inspect diff, commit-tree and non-force push with verified remote receipt. Codex author identity matches historical publication commits. Working-tree HEAD and unrelated files stay untouched.
- `test_archive_safety.py`: 7 allowlist tests pass: metric text accepted, checkpoints/raw arrays/secret/binary/path escapes rejected.

Initial manual snapshot `CHECK_20260929T155302Z.json` had empty ECC dictionaries due to driver-specific XML layout; corrected and superseded by `155359Z`. Preserve the supersession note. The corrected monitor parses flat ECC fields and fails closed on missing preferred-GPU health evidence.

## Recovery and archive contract

Record failures/stalls/PID reuse; inspect only explicitly registered owned jobs. Coordinator directs recovery after checking exact process identity, failure cause, resumable checkpoint/configuration, healthy unoccupied physical GPU UUID, and preserving attempt records. No GPU reset, no broad kill, no automatic blind retries, no unrelated-job mutation.

After each round, mirror locally authored records into the server namespace, finish an explicit allowlist, package server files, download and verify to D: archives FIRST, then prepare a separate Git index from verified publication ref, inspect staged names/content/sizes, commit and push without force. Never upload checkpoint/weights/optimizer/raw data/cache/credentials or prediction arrays. Root authorizes round00 publication after independent prelaunch review PASS; branch `codex/single-model-20260929` based initially on verified `origin/main` commit69bf39f. Preserve unrelated local changes and publication working tree.

## Pending

1. Midnight cron verified; include trigger and app heartbeat receipts in round00.
2. Receive preflight and independent review, record reviewed final source hashes.
3. Mirror all handoffs/coordinator and ops into server round00; complete archive/download/staged inspection/commit/push.
4. Register launched owned pilots from launch receipts; initial smoke validation and durable four-hour checks.
