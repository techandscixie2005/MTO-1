# Monitoring and archive handoff

Updated 2026-09-30 Asia/Shanghai. Owner /root/monitor; goal lifecycle belongs to root. Server scope: /home/inspur/MTO-1/research/single_model_20260929.

## Current run and next action

Round03 is scientifically COMPLETE and awaiting only the reviewed archive/publication workflow. Integrity and scientific reviews passed. The held-out-source affine reaches pooled raw-f validation R2 0.416374515 versus in-sample0.402024351 and native0.405294124; incumbent historical validation-fit0.418119244 remains retained. All labels are preserved; validation is historically reused and no test/independent-seed confirmation occurred. No new fit or inference is authorized by this handoff. Root decision is current_state/ROUND03_DECISION.md; final report is round03_transfer/SINGLE_MODEL_RESEARCH_REPORT.md.

The scheduled heartbeat at server 2026-09-29T19:49:27Z found the registered source healthy at epoch28. The requested terminal follow-up at 19:59:16Z independently verified all33 epochs, source process exit, exact manifest/review/publication binding, and actual checkpoint hashes. Receipt: monitoring/ROUND03_HEARTBEAT_20260929T195916.json; prior observation: monitoring/ROUND03_HEARTBEAT_20260929T194927.json. Final SHA256 769026edf4e999799dd201c3e8f2ad3393e25ef9b3a5a99b23f7ce59ffc984ec; last SHA256 52f2884b4869ae87d85ba56875021ff3d42383cd53bfad356362d213883499c9. Both match FIT_COMPLETE. Science retains sole affine execution ownership. Monitor loaded no tensors, acquired no stage lock, executed no inference and sent no process signals.

Completed source identity: PID1174422, start_ticks1255561468, registry round03_source33_1790707949921845597, physical GPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800, cwd round03_transfer. Source used96,284 fitting molecules, fresh initialization, fixed33epochs and TRAIN-only progress with no validation selection. Initial startup evidence remains ops/ROUND03_STARTUP_CHECK.json and monitoring/CHECK_20260929T185538Z.json. The resumable last.pt and fixed source_final.pt stay server-only; no best.pt is selected.

Resource discrepancy preserved: the terminal snapshot briefly included MTO Python PID1186535 using504MiB on physical GPU0 beside unchanged SpecGPT712998 and Xorg7104. A follow-up ps/nvidia-smi read found1186535 absent and only the preserved SpecGPT compute process remaining. No contemporaneous argv/start/cwd establishes complete fit placement; allocation size alone does not establish incidental initialization, model inference placement or harm. Science retained the frozen fit without repeating it. The reviewed validation launcher then bound GPU1 UUID before Python startup, observed owned PID1187988 only on GPU1, exited0 and preserved coefficient/export hashes. See round03_transfer/GPU_PLACEMENT_REVIEW.md, RESOURCE_PLACEMENT_NOTE.md and VALIDATION_LAUNCH_20260929T200405597244Z.json.

All source/affine commands in ops/ROUND03_RUNTIME_HANDOFF.md have finished. Existing coefficients and completed validation are immutable; do not refit or reopen them. Both affine maps deploy the same full baseline with one readout, one checkpoint each; no averaging. Monitor owns final server-record packaging, FIRST D download/hash verification, exact staged inspection, commit/non-force push and remote verification. Preserve current research-branch ancestry1b34e839 or its verified descendant. No later model fit is automatically authorized.

Frozen Round03 manifest 93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37; review a6807aae55c59f254f2048b1d696a754f130075e32ef4763dfa1c94f4fb1c04c. Server launch receipt ops/ROUND03_PUBLICATION_RECEIPT.json SHA1b28241ea82319087cc6681c0d36caf18a837ee8fc3b84fc6287429d0174df37 binds both.

## Persistent schedules and safety

Canonical read-only user cron remains 0 */4 * * *, running ops/monitor.py at 00, 04, 08, 12, 16, 20 server Asia/Shanghai. Actual midnight trigger previously verified. Generic status/log/checkpoint-stat monitoring supports 33 epochs and training-only fields; no sealed monitor changes were necessary.

Existing app heartbeat qm9s-e-a remains ACTIVE every 4 hours, thread 01a0edd8-8c03-75d1-aeb0-953f11af6433. Updated via app tool after publication to read current COORDINATOR and ROUND03_RUNTIME_HANDOFF, verify source completion/exit/hashes, coordinate the authorized idempotent affine stages, and preserve archive-first ordering. Name, cadence, target, creation time and notification policy were independently verified unchanged; quiet unless meaningful progress/failure/completion/action. Evidence ops/ROUND03_HEARTBEAT_VERIFICATION.json and HEARTBEAT_BEFORE_ROUND03.toml / HEARTBEAT_AFTER_ROUND03.toml. These supplemental records belong in the next archive, not the immutable preparation bundle.

Registry ops/registry.json pins PID/start_ticks/bootID/UID/cwd/argvSHA/GPUUUID and scope-contained paths. Recovery requires actual owned failure, identity and resumable-state checks plus coordinator direction. Never infer failure from agent/model interruption, missing observations, or expected preterminal files. No blind restart, broad kill, GPU reset or unrelated mutation. The server clock was measured about 137 seconds behind Windows; use host-labeled times and receipt dependencies.

GPU0 unrelated SpecGPT PID 712998/supervisor 712791 remains preserved. Exclude unhealthy 3/7 (uncorrectedECC/pendingremap); GPU5 is reserve. Preferred GPUs 1/2/4/6 require fresh idle/ECC/remap/UUID checks and /tmp/mto_pouter_gpu_INDEX.lock. UUID2 GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa;4 GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184;6 GPU-2431a641-8045-a10b-4aa0-2769010e0708.

## Verified archives and ancestry

All archive directories are under D:\MTO\archives\single_model_20260929 and immutable after publication.

| Archive | Remote verified commit | Parent |
|---|---|---|
| round00_audit |0e612523cd99594b6ce05b491b8de1b18d8a2e66|69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e|
| round01_factorial_complete |2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae|0e612523|
| round02_frozen_preparation |a4c2fabd837bea5ae7eced79d11de797af3900bf|2589d0f|
| round02_frozen_complete |bdc28c351b941a58dae49f833284e13cf5fdc46b|a4c2fabd|
| round03_transfer_preparation |1b34e839335177e1fd582944549be4f181dee3c0|bdc28c35|

Round03 prep: 90 server files, 505,477 bytes, 92 byte-exact staged blobs; bundleSHAb55213509aee9477d9d0e4bccc2c6176dea2326b6c09b58979e7a5d3918aa990. Includes all 28 top-level science records, 62 lightweight closure snapshots plus 7 server-only data/identity/split/checkpoint hash references, authorization and runtimehandoff. Round02 complete: 177 server files, 1,968,909 bytes, 179 blobs; bundleSHA51fab0cfb494bcf2a80391a648b318a9b1d791497e709f3ae2748c7eee622326. Both adapters were null; strongest eligible calibrated eta0 remains server baselines/calibrated_eta0.pt. No test evaluation or independent-seed promotion was justified.

## Publication procedure

Finish scientific review and root decision, mirror local lightweight records server-side, inspect explicit allowlist, package, FIRST download to a new D archive and verify every hash, THEN stage exact bytes in separate index, inspect content/names/sizes, commit/non-forcepush and verify remote. Never copy/upload weights, optimizer tensors, raw data, identity or split arrays, predictions, caches or credentials. Metadata/hashes only for checkpoint provenance. Failed/aborted rounds use ops/FAILED_ROUND_ARCHIVE_POLICY.md; never fabricate completion or omit failures.

Publication Git D:\MTO\publication\MTO-1 has an unrelated unborn main worktree and preexisting untracked PUBLICATION_TREE_IDS.json; preserve both and dirty C: workspace. Reuse ops/PUBLISH_COMMAND.ps1 (absolute native Windows OpenSSH through scoped CONNECT to ssh.github.com:443, existing identity, strict known-host checks). Do not repeat failed direct transports or change global configuration. hash-object --no-filters; write-tree --missing-ok only for unchanged historical objects; all new blobs independently byte-verified by verify_staged_records.py. Next parent must be current remotely verified research branch codex/single-model-20260929 (now 1b34e839 or descendant), never rebase on main. Copy derived publication receipts server-side for the next archive.

## Pending

No additional experiment polling is needed: source and affine stages are complete. Four-hour schedules remain active, with quiet behavior unless a meaningful authorized action exists. Finish current completed-round archival using the exact independent final manifest and frozen root decision; collect operational/heartbeat/startup records and current handoffs. Preserve every prior archive and all server-only checkpoints/optimizer/index/prediction arrays. No new fit or validation rerun follows automatically.


