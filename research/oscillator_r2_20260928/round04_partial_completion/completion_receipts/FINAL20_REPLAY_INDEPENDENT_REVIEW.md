# Independent review: deferred final20 replay

Date: 2026-09-29. Verdict: PASS for preparation; no blocking implementation issue found. This review does not authorize execution. All four architecture arms, the round summary and coordinator authorization must precede any replay. No active-run status or GPU occupancy was inspected during this review.

## Reviewed hashes

- final20_replay.py: 41d3259d814579f5500a02c73d84a44009b1cda4d8b21de7eceb217a2de84beb
- FINAL20_REPLAY_PROTOCOL.json: 5ead8356bfe88b66edf932c193626d6921664e227acc1a19f755df5462165c9f
- FINAL20_REPLAY_README.md: 228a7520c529405ad8ffecd0858ed696f61f0cae68c77467b2bb24b5c31ac143
- FINAL20_REPLAY_SELF_TEST.json: 451cc97b19ca4683597628a1b80c3d6c84bf21ef05ec2a694769e00855308bb5

The recorded CPU self-test is bound to the reviewed script and protocol. I additionally imported only the helper and exercised summarize on synthetic arrays with all eight state/brightness/sign cells nonempty. Partition counts, direct SSE decomposition, absolute-fold identity and top-k capping passed. That independent check did not instantiate ValidationInputs, import the model or torch, open scientific data/checkpoints, use CUDA, or write result files. Runtime inference has not been verified by this review.

## Scientific arithmetic and alignment

The primary base is recomputed as C_F * E_pred.double() * trace(base_A.double()), then delta_f.double() is added before abs. This matches architecture/objective.py native_f64. Float32 forward quantities are separate sensitivity columns; they do not silently replace the native64 primary metric. Equality against native_f64, finite/shape assertions and stored-final R2/E-MAE tolerances are checked before publication.

The loader uses only frozen validation indices and input rows. Raw targets and masks come from the hash-pinned eta0 validation export whose raw-label alignment was independently established in the completed-arm receipt. It asserts IDs/indices, 6686 molecules, ten f slots, unique IDs and all-valid masks; unsupported masked data would fail rather than be silently included. Model batches contain only z, pos, batch, n and edge_index. True E and f are used for diagnostics only, never as oracle inputs.

Full dataset/raw-label archive hashes are computed bytewise for integrity. This does not decode or consume test labels for inference or selection. The loader never accesses dataset E/A or full raw target arrays, and never reads test indices or prediction files. Eager validation-input preparation reads all validation features/targets into memory; only two feature rows were batched in the recorded self-test. The description should not be interpreted as limiting validation target loading to two rows.

The eight-cell cross-table partitions physical S7/S8, true-f threshold .0546 and native64 signed output. It retains negative contributions and permits shares above100% when other cases improve. Top-1/5/10/1% concentrations are molecule-level, sum all ten states and use stable ranking; the validation top1% has ceil(.01*6686)=67 molecules. Base-minus-eta0 plus final-minus-base reconstructs total degradation. The fold identity correctly compares emitted abs output to the unphysical signed output and is not evidence that folding caused harm relative to eta0.

## Provenance and deferred execution gates

Separate authorization binds script, protocol, final checkpoint and chosen admitted GPU. The helper requires round-summary existence and all configured FIT_COMPLETE markers. It verifies pinned checkpoint/completion/preflight/review/data/eta0-export hashes, agreement of the sealed source manifests, each source digest, residual completion/selection epoch and absence of residual failure/invalidation markers. The sealed manifest includes model/objective, initial weights, original checkpoint, stats/config, GPU health and admission code/record.

The shared /tmp/mto_pouter_gpu_<gpu>.lock is held across GPU health checks and replay. Allowed GPUs exclude0/2/3/7. Health admission requires idle process/memory state and acceptable ECC/remap/repair counters. GPU5 additionally checks pinned counters and runs the numerical microcheck before model construction. During/following inference, ECC/UUID stability and absence of foreign GPU PIDs are checked. No preemption or resets occur.

The pre-run existence checks prevent routine repeat execution. Publication writes the array atomically, then the summary. If interrupted between those writes, a partial-publication state will deliberately block a blind rerun; recovery should inspect the server artifact first. The coordinator should issue one execution at a time; this helper is not a general concurrent replay service.

## Scope, limitations and disposition

The README and result limits correctly label this as an inference decomposition of a jointly trained checkpoint, not retraining without the residual, a matched architecture win, or a new selected model. Epoch0 selection remains unchanged. No fitting, optimizer steps or checkpoint mutation occur.

Raw arrays go under completion_receipts/runtime on the remote server; only the small result JSON is designated for later archive. The script contains no transfer/upload action. Archive exclusion remains an operational requirement for the coordinator's later transfer, not something a Boolean result field can enforce.

Error concentration supports distinguishing rare errors from diffuse deterioration. The current summary does not report mean signed prediction error or all signed-error quantiles, so a specific broad-bias claim may need an offline calculation from the saved server array. That requires no second model replay and is not a launch blocker. No helper/config changes requested. No actual replay, checkpoint/model forward, test-label use, GPU invocation or sealed-file modification was performed in this review.