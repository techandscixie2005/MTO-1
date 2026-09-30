# MTO monitoring and archive handoff

Updated after the 2026-09-30T11:53Z scheduled check. This wakeup's delegate: /root/qc_design because /root/monitor could not resume under the thread limit. The existing dedicated monitor and schedule remain in place. Root owns strategy and goal lifecycle; science owns execution; history owns independent review. Server campaign: /home/inspur/MTO-1/research/single_model_20260929.

## Current state

At 11:53:44Z all 17 registered original wrappers were absent: 15 complete markers and two known retained failure markers. The failures are the original split materialization (subsequently repaired and independently verified) and original GPU technical preflight (full-forward equality failure before any optimizer update). All eight newly registered stages' child processes are absent; terminal log and applicable output hashes match, including the original audit's preserved snapshot. No owned GPU process was observed. The eight monitor incidents are newly observed registry transitions (six completions and these two known failures), not eight new failures. No preferred-GPU health incident was detected.

Canonical observation: monitoring/CHECK_20260930T115345Z.json (SHA256 fa227b2dc0abfa979f572b017fe87ee99faebbb5350741f5c17fb1bf7332ec14). Concise receipt: monitoring/SCHEDULED_PREPARATION_20260930T1153.json (SHA256 93919e6f42421c2007bfa0d40a3acd6c0e2f455c0814bf939a47ddbb2a4cf748). All original receipts and failure evidence remain intact. This check performed no fit, inference, weight loading, restart, signal, GPU admission or publication.

Science subsequently reported root approval for the narrow Round05 identity-check amendment and reviewed retry, limited to 12 discarded TRAIN technical updates. The 11:53 observation predates that retry and is not its terminal status. Do not repeat a check until the registered retry terminal review or the next scheduled check. No production fit is authorized by this monitoring handoff.

Round04 is fully complete and published to main and codex/single-model-20260929 at 8497e0ba10efc3226197c894f46507acdad8ec54, parent 74810c5a. Both scalar maps failed promotion; retain the legacy calibrated eta0. No Round04 execution, recovery, validation or archive is pending. Publication receipt ops/ROUND04_COMPLETION_PUBLICATION_RECEIPT.json SHA256 564a1288c7240e82375ae539e250cc6686c188da812e087270331d951ea57ab3. Archive D:/MTO/archives/single_model_20260929/round04_fonly_complete contains 53 server records and 55 exact published blobs. Preserve sealed snapshots and all prior receipts.

## Authority and next work

Read current_state/COORDINATOR.md and RESEARCH_RESUMPTION_20260930.md. The continuing target is single-model pooled raw-f R² 0.60 and remains unmet. The v2 group-disjoint split is privately materialized and independently verified; it is a new partition of historically exposed data. Round05 fresh control/F/raw-M-decorrelation/both preparation is selected; final technical closure/review and root acceptance remain required. The completed QC congruence design is deferred, with synthetic evidence only and no implementation/fit authorization. Root may authorize reviewed in-scope pilots without repeated user approval; this scheduled check authorizes no fit or test scoring. Do not rerun completed stages or tune on test outcomes.

## Persistent monitoring

The existing qm9s-e-a heartbeat was not changed or duplicated by this check. The last schedule verification at 07:55Z found it ACTIVE every four hours on thread 01a0edd8-8c03-75d1-aeb0-953f11af6433; evidence remains ops/HEARTBEAT_CLOSED_ROUND04_20260930T0755.json and before/after TOML snapshots. It stays quiet for unchanged/non-actionable state.

The read-only server cron remains unchanged (SHA256 13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e). Registry ops/registry.json pins PID/start/boot/UID/cwd/argv/GPU identity. CPU wrappers use ops/register_cpu_stage.py with gpu_uuid=null and a hidden CUDA environment; GPU registrations use the existing GPU registrar. Never infer failure from stale PID or missing observation. No blind restart, broad kill, GPU reset or unrelated modification.

Preserve GPU0's unrelated SpecGPT job. GPUs3/7 remain excluded for ECC/remapping; GPU5 is reserve. Preferred1/2/4/6 require fresh healthy/idle admission, physical UUID binding before imports and /tmp/mto_pouter_gpu_INDEX.lock. The former Round03 affine-fit placement uncertainty remains in archived evidence; do not reinterpret or repeat that fit.

## Publication

/root/qc_design holds as prospective Round05 preparation publisher. No package, D archive, staging or push before final preparation acceptance. Follow round05_scratch_preparation/PROSPECTIVE_ARCHIVE_CHECKLIST.md; preserve both split attempts, CPU/GPU preflight failures, zero-update diagnosis and reviewed amendment/retry evidence. Exclude all private_partition and private_preflight* directories. Capture exact lightweight dependency sources or byte-exact restoration mappings. The completed split has a reviewed 89-record lightweight manifest; final Round05 closure is still pending.

After scientific review and root decision, FIRST download lightweight records to D:/MTO/archives/ and verify hashes; THEN inspect exact allowlisted Git bytes, commit/non-force push and verify remote. Next ancestry is 8497e0ba or a verified descendant. Reuse ops/PUBLISH_COMMAND.ps1 with scoped SSH transport and separate index; preserve dirty worktrees and default index. Never publish weights, coefficients/tensors, optimizer states, raw data, identities/split arrays, predictions, caches or credentials. Keep resumable states on the server. Failed rounds retain honest logs/decisions under the existing failed-round archive policy.
