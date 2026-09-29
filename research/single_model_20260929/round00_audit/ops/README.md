# Four-hour monitoring and archive operations

Scope: `/home/inspur/MTO-1/research/single_model_20260929`.

`monitor.py` runs from user cron at 00:00, 04:00, 08:00, 12:00, 16:00, and 20:00 server time (Asia/Shanghai). It records `monitoring/CHECK_*.json`, atomically updates `monitoring/latest.json`, and appends a small event journal. A file lock prevents overlap. The existing Codex heartbeat `qm9s-e-a` is ACTIVE every four hours on current chat `01a0edd8-8c03-75d1-aeb0-953f11af6433`; it dispatches a dedicated monitoring agent for diagnosis, scoped recovery, archiving, and strategy continuation. The server cron performs read-only machine checks independently of the desktop.

## Ownership and recovery

Register each launched experiment with `register_run.py`. It records PID, process start ticks, boot ID, owner UID, working directory, command SHA256, GPU UUID, progress paths, terminal markers, and checkpoint metadata paths. All artifact paths must lie inside the owned research scope. Use a new attempt ID for any resumed process.

The monitor never signals or restarts processes. Missing workers, PID reuse, stale progress, explicit failure markers, and newly unhealthy preferred GPUs are recorded for coordinator review. Before resuming, the monitoring agent must check the launch receipt, exact process identity, failure cause, readable resumable checkpoint and its training configuration, and healthy unoccupied GPU UUID; preserve the failed attempt. Never restart an already live worker. Never reset a GPU or touch unregistered jobs. Resume only a coordinator-authorized owned run with its recorded command and checkpoint.

Preferred physical GPUs are 1, 2, 4, 6. GPU 0 is occupied by unrelated SpecGPT. GPUs 3 and 7 have uncorrected ECC errors and pending row remapping and are excluded. GPU 5 is reserve because of corrected ECC history. Recheck ECC, remapping, compute-process occupancy, memory, and temperature immediately before launch. UUIDs and audit values are in `registry.json`; index alone is insufficient.

## Verification and known superseded snapshot

`MONITOR_SAFETY_TESTS.json` records read-only tests of process identity, PID reuse, path containment, completion/failure handling, and actual GPU-health parsing. The initial manual record `CHECK_20260929T155302Z.json` used an incompatible ECC XML field path and is superseded by `CHECK_20260929T155359Z.json`; it is retained as provenance. The corrected monitor parses this driver's flat `ecc_errors/volatile` and `aggregate` fields, and treats missing expected health fields as requiring review for preferred GPUs.

## Archive before publication

1. Finish the round report, exact settings, source hashes/changes, logs, results, analysis, and next decision on the server. Mirror locally authored coordination documents into the round records first.
2. Write an explicit relative-path allowlist. Run `package_records.py --allowlist <file> --output <bundle> --round-id <id>`. This rejects binary formats, dangerous paths, checkpoints, weights, raw datasets, caches, optimizer-state artifacts, likely credentials, files above 5 MiB, and bundles above 25 MiB. Review the selected filenames and contents; automated checks are supplementary.
3. Run local `download_records.py --remote-bundle <bundle> --expected-sha256 <printed hash> --round-id <id>`. It downloads to `D:\MTO\archives\single_model_20260929\<id>`, verifies the bundle and every member hash, validates paths, extracts records, and writes `LOCAL_DOWNLOAD_RECEIPT.json`. Existing archive directories are never overwritten.
4. Only after this receipt exists, stage exactly the reviewed downloaded records into the publication Git repository `D:\MTO\publication\MTO-1`. Inspect staged paths, sizes, and content before commit/push. Preserve its preexisting `PUBLICATION_TREE_IDS.json` and the unrelated dirty local workspace.
5. Record publication commit, remote branch and verified remote commit ID beside the archive. No checkpoint or model/optimizer state, raw data, caches, credential, or unreviewed file is eligible for upload.

The publication repository has an unborn working-tree HEAD and existing remote refs; use a separate Git index plus `read-tree`, `hash-object`, `update-index --cacheinfo`, `write-tree`, and `commit-tree` based on the verified research parent ref. Inspect the exact diff before creating/pushing a research branch. Do not check out its historical data-heavy tree.
