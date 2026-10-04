# Round09 preparation status at resource hold

CPU engineering passed; GPU engineering remains unexecuted because physicalGPU1 failed the fixed idle-memory admission check. This is not a model, optimizer or data failure and provides no accuracy evidence. Full training remains unauthorized.

## Completed and reviewed

- Exact accepted original-MTO coupled AdamAMSGrad1e-4 versus0 implementation, one ordered135-tensor/1,552,092-element group; all original trainables included,11 frozen tensors excluded. No parameter-group or coefficient tuning.
-22 synthetic production authorization checks passed before scientific imports/artifacts.
- CPU_SOURCE_REVIEW SHAca35f1c8ccfc9996032e72bab0cd7bec654290081de7e8b0951ec3973b4e21cd,295 source/dependency pins.
- CPU_PREFLIGHT SHAdfca79770fa451a64e08bd936b80294301829d527a10e60a48ccfe0e98d854c5 passed once. Five toy Adam steps verified exact coupled-afterclip semantics, moments/AMSGrad history, None-versus-zero behavior; maximum arithmetic error2.22e-16. Zero full-model optimizer updates. Fresh model identity/task gradients, synthetic mask/symmetry and one-file export checks passed.
- PARAMETER_ROSTER SHA290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943 was frozen before all optimizer calls and matches the unwrapped original roster exactly.
- CPU result reviewc2147eb3 passed. TECHNICAL_SOURCE_REVIEW SHA5c4b82c015ec2b5abc475fbaa846757f7eb5b3fc9e0bc00991bacef0175ed457 binds311 files for exactly six discarded first128 TRAIN updates. Two earlier metadata-binding checks found missing identical helper copies; they were restored and failures retained, with no numerical rerun.

## Actual resource rejection

WrapperPID1863309 exited1 at resources.admit(1), memory_used<1000MiB assertion. The attempt directory was created, but no child process, registration, CUDA model, data decode or optimizer update followed. ops/GPU_ADMISSION_FAILURE.json records absent private_preflight/GPU_PREFLIGHT/GPU_UPDATE_PROGRESS and the empty preserved attempt. The later read-only snapshot observed unknownGPU1 PID1860829 using2248MiB; it was not signaled. GPU2 was also busy. Snapshot occupancy is historical and grants no future allocation.

Original sources/review/log/XML were copied byte-for-byte into admission01_preserved/MANIFEST.json SHAab9c820e719b241021879d786a5b78f04d95f103fe0e941344ea7e4592529e65. No automatic retry/fallback or settings change. A future distinct owned attempt requires root's resource/retry decision and independent operational review. Do not delete/reuse the original empty attempt.

## Remaining gates and limits

1. Explicit resource/retry decision; then reviewed distinct attempt with fresh health/UUID/exclusive-lock/registration barrier.
2. The already fixed six discarded TRAIN updates, replay/access checks and independent actual-result review. No additional update budget or VAL/TEST access.
3. Metadata-only freeze/final independent preparation review, root acceptance and D-first publication.
4. Separate bound production authorization for any60-epoch fits.

PROTOCOL.md and RUNNER_HANDOFF.md give exact future settings, blocked commands, authority schema and recovery semantics. freeze_preparation.py is prepared but not executed; no FROZEN_MANIFEST or final preparation PASS exists. The old proposal manifest remains unchanged, with final preparation supplement reserved for a distinct filename.

Engineering checks do not establish empirical performance, generalization or an optimal decay scale. Toy CPU optimizer agreement does not replace CUDA replay. The first128 TRAIN fixture will not guarantee full-batch memory needs. The0.60 target remains unmet; retained validation reference is.44716940136585204, without independent seed/TEST confirmation. All weights/arrays remain server-only; no new validation or TEST values were opened.
