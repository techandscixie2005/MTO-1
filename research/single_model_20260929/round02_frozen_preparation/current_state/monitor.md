# Dedicated monitoring and publication handoff

Updated 2026-09-30 Asia/Shanghai. Owner: /root/monitor. Server scope: /home/inspur/MTO-1/research/single_model_20260929.

## Current state

Round01's four fits completed all 20 epochs normally. All selected epoch0, without native pooled raw-f improvement. Summary, mechanism/gap audits, aggregate plot and false-bright replay completed. Independent integrity/scientific review PASS and root decision are archived. The science agent's administrative interruption did not affect server jobs; no fit restarted.

Round01 archive: D:\MTO\archives\single_model_20260929\round01_factorial_complete. Its 163 server files total 2,462,550 bytes; bundle SHA256 f95858ef2c0b4be769a9bd667c7d7f1c4c9e6f2eff3442c8d780979d7223a330. Download and all hashes verified before staging. All 165 new Git blobs were byte-verified. Remotely verified commit 2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae, parent 0e612523cd99594b6ce05b491b8de1b18d8a2e66, branch codex/single-model-20260929. Server receipt ops/ROUND01_PUBLICATION_RECEIPT.json. Bundle is immutable.

Round02 source frozen and independently reviewed PASS. Manifest SHA256 6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e. Review SHA256 38e0d09ad8d007c5ee7036a9da5fc279ec63019bbca84ee25a0cc8774f4f6452. Two frozen-base F arms compare LE+Ls and LE+normalized raw-f with coefficient1. GPU2 bounded TRAIN cache/parity/resume tests passed and lock released. No production Round02 fit launched yet. Cache remains server-only. Root conditionally authorizes launch after new archive-first publication receipt, immediate healthy GPU/lock admission and registration. Next publication must descend Round01.

## Durable monitoring and ownership

Server cron remains 0 */4 * * * running ops/monitor.py with root scope; logs under monitoring/. Actual midnight trigger verified in monitoring/CHECK_20260929T160002Z.json. Cadence is 00,04,08,12,16,20 server Asia/Shanghai. Existing app heartbeat qm9s-e-a ACTIVE, FREQ=HOURLY;INTERVAL=4, targets thread01a0edd8-8c03-75d1-aeb0-953f11af6433. No duplicate schedules.

Registry ops/registry.json pins PID/start_ticks/boot_id/UID/cwd/argvSHA256/GPUUUID and scope-contained paths. New attempts require distinct IDs. Monitor is read-only and records checkpoint stat metadata only. Recovery requires coordinator direction, exact ownership, actual failure diagnosis, resumable state and healthy resource admission. No broad kill, GPU reset or blind restart. Never infer training failure from administrative model error or missing observation.

Final Round01 snapshot monitoring/CHECK_20260929T171902Z.json has all four complete. Original compact-observer terminal race was preserved and superseded; no intervention occurred. Canonical monitor reads terminal markers after identity sampling.

Preferred physical GPUs and UUIDs:
- 1: GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800
- 2: GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa
- 4: GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184
- 6: GPU-2431a641-8045-a10b-4aa0-2769010e0708

Always acquire shared /tmp/mto_pouter_gpu_<index>.lock and check fresh idle/ECC/remap/UUID data. GPU0 has unrelated SpecGPT PID712998/supervisor712791: never touch. GPUs3/7 have uncorrected ECC/pending remap: exclude. GPU5 reserve.

## Archive and publication

Mirror final handoffs to server, inspect explicit lightweight allowlist, package, FIRST download and hash-verify to a new D:\MTO\archives\single_model_20260929 directory, THEN stage exact bytes in separate Git index, inspect content/names/sizes, commit, non-force push and verify remote. All weights/optimizer/data/split membership arrays/predictions/caches/credentials stay server-only. Metadata/hashes allowed. Failed rounds use ops/FAILED_ROUND_ARCHIVE_POLICY.md, never fabricate completion or omit failures.

Publication Git D:\MTO\publication\MTO-1 has preexisting unborn main worktree/untracked PUBLICATION_TREE_IDS.json. Preserve it and unrelated C: dirty files. Use ops/PUBLISH_COMMAND.ps1 and verify_staged_records.py. Native Windows OpenSSH through scoped ssh_proxy.py CONNECT to ssh.github.com443 uses existing SSH identity and strict known-host checks. No global config changes or credential logging. Direct transports/HTTPS credentials already failed; reuse working transport.

hash-object --no-filters preserves bytes; write-tree --missing-ok preserves unchanged historical partial-clone objects. All new blobs must exist and match archive bytes. Parent must be remotely verified current research branch, never fresh main rebase. Publication receipts copied server after push.

Round00 remains round00_audit, commit0e612523cd99594b6ce05b491b8de1b18d8a2e66, parent69bf39f. Frozen scientific manifest eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3. Eligible calibrated eta0 remains server baselines/calibrated_eta0.pt; never upload. Server clock was ~137s behind Windows; host-labeled times and receipt dependencies establish sequence.

## Next actions

1. Archive/publish Round02 preparation with pinned source/dependency mapping and exact independent review.
2. Supply manifest-bound receipt to science for conditionally authorized two-arm launch; science owns launch/registration.
3. Independently verify startup identities/progress/checkpoint metadata once, preserving four-hour cadence.

