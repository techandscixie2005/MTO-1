# Dedicated monitoring handoff

Updated: 2026-09-30 00:28 local Asia/Shanghai. Active research scope `/home/inspur/MTO-1/research/single_model_20260929`.

**Latest operational state (2026-09-30 ~01:24 server time):** All four original fits completed their fixed20epochs and exited normally. All selectedepoch0; no native raw-f improvement. Administrative science-agent model-capacity failure did not affect server jobs; no job was restarted. Monitor took over and successfully executed reviewed CPU-only `summarize.py`, `mechanism_audit.py`, `gap_audit.py`, `plotting.py` sequentially. Outputs/logs exist; history independent review and root decision are pending. Science owns one additional root-authorized false-bright selected0-vs-fixed20 replay on admitted GPU1; do not duplicate it. All raw arrays/checkpoints remain server-only. Final archive/download/publication has **not** occurred yet and must wait for review and decision.

Final authoritative monitor record is `monitoring/CHECK_20260929T171902Z.json` (four complete transitions). Compact observer `LIVE_CONTINUATION_CHECK.json` initially sampled adapter after its normal exit and misclassified missingPID; that observer-only race is corrected and the original retained as `LIVE_CONTINUATION_CHECK_before_terminal_race_fix.json`. Canonical cron reads terminal markers after `/proc` and never restarts any worker. Corrected observer distinguishes natural terminal absence from absent-without-terminal; no operational intervention occurred.

Next archive inventory also includes explicit confirmation-preparation and report/cache manifest files, full diagnosticmetadata originals/amendment, DECISION_RULES, and the exact text SVG `ALIGNED_VALIDATION_CURVES.svg`. `FAILED_ROUND_ARCHIVE_POLICY.md` documents a separate failed/aborted archive route; the successful20epoch gate does not suppress actual failure records. `verify_staged_records.py` checks each downloaded byte against staged Git blobs and prohibits surprise changes; subsequent rechecks use unique receipt names. `publish_records.py --publish` now requires a matching passed CONTENT_REVIEW. Use native Windows OpenSSH through scoped CONNECT, `--no-filters`, and `write-tree --missing-ok`; next parent must be the remotely verified research branch0e612523 or its descendant, not a freshmain rebase.

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

## Completed publication and startup

Round00 is archived at `D:\MTO\archives\single_model_20260929\round00_audit`: 83 server records, 405187 bytes; bundle SHA256 `9074a4ae97af81d8dbeb274f1506980dfd9925f4a0367f151756fa2b36b11549`. The downloaded files plus manifest/download receipt formed 85 explicitly staged blobs, all independently byte-verified against the D: files. Largest file is 30545 bytes; total 420282 bytes. No binary model/data artifacts. Commit `0e612523cd99594b6ce05b491b8de1b18d8a2e66` was non-force pushed and independently verified on branch `codex/single-model-20260929`, parent `69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e`. Scientific source manifest is `eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3`. Server launch gate receipt: `ops/ROUND00_PUBLICATION_RECEIPT.json`. The unrelated publication working tree remains untouched.

Publication repair is recorded for the next round: direct SSH22/SSH443/HTTPS443 failed; scoped system HTTP proxy enabled HTTPS reads but no HTTPS credential existed. Native Windows OpenSSH through the scoped Python CONNECT relay succeeded using the existing GitHub key and strict known-host verification. Use **absolute `C:/Windows/System32/OpenSSH/ssh.exe`** in `GIT_SSH_COMMAND`; bundled Git SSH incorrectly invokes PowerShell with a Unix `exec` prefix. No global config or credentials were changed. Use `write-tree --missing-ok` to retain unchanged historical partial-clone objects and `hash-object --no-filters` to preserve exact downloaded bytes. The first filtered staging attempt was rejected before commit. Old cancelled `.publication-index` locks were renamed after checking no publication process remained.

The revised helper is local/server `ops/publish_records.py`; revision SHA256 `da4f5aae74cb72a2fcd2cf222654d6ce0da874c068e8adefb33e77a60df6beb6` is downloaded at `round00_transport_helper/publish_records_v3.py`. The initial round archive is immutable. Collect supplemental transport helper/receipts in the completed-fit archive.

Four owned pilots are running and independently identity-verified:

| Arm | PID | Physical GPU |
|---|---:|---:|
| control |1132183|1|
| adapter |1132188|2|
| decorrelation |1132193|4|
| both |1132198|6|

Startup snapshot `monitoring/CHECK_20260929T162421Z.json` verifies four identities, exact GPU UUID assignment, healthy preferred hardware, and unrelated GPU0 PID712998 preserved. Four incident entries in that snapshot are first-observed RUNNING transitions, not failures. `ops/INITIAL_OWNED_CHECK.json` records a single targeted CPU checkpoint-schema audit: all four epoch0 metrics, best.pt and last.pt exist, last includes model/optimizer/Python+NumPy+Torch+CUDA+order RNG/config/manifest. No weights were downloaded. Recurring monitoring only stats checkpoint metadata and never loads weights. Next cron slot remains04:00 server time; startup checks do not reset cadence.

Clock caveat: the server clock was measured137.07 seconds behind local Windows at local epoch1790699264.07 (round-trip467ms). Server launch timestamps can appear before local publication timestamps even though the launch script read the published and remote-verified receipt first. Use host-labeled timestamps plus receipt dependencies to establish sequence; no clock was changed.

## Pending

1. Keep the four-hour server cron and existing app heartbeat active. Diagnose only owned failures; no blind resume or unrelated intervention.
2. On round completion, combine reviewed fit records with supplemental ops/transport records, current coordinator/handoffs, and response-feasibility lightweight records. First mirror local records into server scope, then download to a new D: archive, inspect, commit and push.
3. Use the preserved one-checkpoint calibrated eta0 baseline and frozen baseline path for comparison; never upload either checkpoint or prediction arrays.

## Authoritative continuation check and completed-round inventory

Root requested one live continuation check. `monitoring/CHECK_20260929T162842Z.json` sampled `/proc` plus actual NVIDIA process/GPU/health data; `ops/LIVE_CONTINUATION_CHECK.json` reverified `/proc` identities at server00:29:38 and read complete epoch records. All four had completed epoch2, no failure marker, no terminal marker, and best epoch0. Interim latest native R²: control0.4008791, adapter0.4007813, decorrelation0.4008805, both0.4007831. Epoch times143.5–161.2seconds imply roughly43–48minutes remaining at that observation. No intervention or scheduler change. These are interim observations; finish the fixed protocol.

`ops/prepare_completed_round.py` and `ops/COMPLETED_ROUND_INVENTORY.json` now define the prospective completion archive. Inventory status is explicitly `prospective_only_not_a_completed_round`:94 existing lightweight records, with four FIT_COMPLETE records and two ROUND_RESULTS reports pending. The finalizer requires all four20epochterminal records with one checkpoint/no test, no exact owned worker still live, independent PASS review, and coordinator scientific decision. It includes source/settings, every arm's logs/metrics/launch/registration/admission records, monitoring, transport fixes/publication receipts, current handoffs, and response-feasibility source/results. It rejects binary checkpoints, arrays, caches and credentials.

After completion and review: `python ops/prepare_completed_round.py --finalize --review <review.json> --decision <decision.md> [--extra <record>]`; inspect `ops/COMPLETED_ROUND_ALLOWLIST.txt`; package that list to a new round ID; download/verify to D: FIRST; stage and byte-review; commit/nonforcepush/remoteverify. `ops/PUBLISH_COMMAND.ps1` records the tested native-OpenSSH/CONNECT environment for the existing SSH identity; use it once to prepare and again with `-Publish` only after staged inspection. It makes no persistent Git/SSH configuration change.
