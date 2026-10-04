# Round06 fixed technical preflight boundary

Root bounded preparation decision d8f7478f77bf660c32e75897d555b13f972e64abaac4385e2c9b44ab3e2e6458 authorizes implementation, synthetic checks, and (only after independent exact source review and fresh GPU admission) six discarded TRAIN updates. Production remains unauthorized.

## Synthetic loss and precision checks

Run CPU-only with CUDA_VISIBLE_DEVICES empty before imports and OMP/MKL threads2. No target/prediction arrays or existing weights are read. Hand-set energy/PSD/target fixtures cover both objective definitions, the exact eV→Hartree conversion, original control bitwise equality, printed-f zeros, and invalid-mask sentinels selected before nonlinear product/error arithmetic. Check candidate gradients with analytical partial derivatives for E and A, including small and zero strengths. Training is FP32; expected analytical gradients are evaluated independently in FP64 from FP32 input values. Fixed gradient tolerance: atol2e-6, rtol1e-5. Mathematical synthetic O(3) loss covariance uses the same tolerance; no change after outcomes. Reject empty valid sets, wrong objective, wrong normalization and wrong training dtype explicitly. No finite cap or replacement is permitted.

Inherited original backbone/disabled-F O(3), permutation/translation, constructor and one-checkpoint access checks remain bound by Round05's source/preflight. Reuse exact readers/runtime/predictor where unchanged; document their byte hashes rather than repeating unrelated numerical probes. A fresh synthetic CPU initialization confirms both arms share the prior initial base/full hashes and dormant F stays frozen. No trained checkpoint is loaded.

## GPU fixture — exactly six optimizer updates

Use exactly the first128 ascending new TRAIN rows, the same metadata selection as Round05, split into consecutive batches64. The raw target reader checks rows/fields before opening each archive and logs actual numeric decode indices. No validation or TEST fixture is permitted.

For each of trace_control and raw_f: rebuild fresh seed11; check prior base/full initial hashes; execute update1 on first64, save disposable model/Adam/RNG/order state, execute update2 on second64, reload the update1 checkpoint on CPU, and replay update2. This is3 actual optimizer updates per arm,6 total. No additional train update is authorized. The technical namespace is private_preflight and may never seed production or enter publication.

The first batch also supports a zero-update gradient audit using retained graphs: LE and each intensity term versus shared parameters, direct d(intensity)/dE and d(intensity)/dA, and active energy/PSD-head parameter gradients. Candidate direct E/A intensity gradients must be finite/nonzero on the actual fixture; control Ls has no direct E-output derivative, which is expected. Record magnitudes and ratios, not a gradient-normalized new objective. All actual updates call the production objective function, fixed AMSGrad LR.001, clip5, FP32. Report each preclip norm/clip flag, losses, timing and peak memory. Small early TRAIN rows do not establish worst-case memory or full-run performance.

Fixed inherited numerical criteria: model/optimizer resumed-update atol2e-6, rtol1e-5; loss replay atol1e-6, rtol1e-5. RNG/order/integer states and initialization hashes are exact. Full GPU forward/geometry-export E/A parity uses the explicitly accepted prior post-diagnostic atol2e-6, rtol1e-5, with absolute f differences reported. CUDA execution variability is not described as whole-trajectory bitwise determinism. No outcome-dependent tolerance adjustment.

Use only the first two already decoded fixture geometries for original-versus-wrapper and exported-geometry parity. Check a one-checkpoint loader under an access guard during construction and forward; source and exact nonpersistent buffer fingerprint remain strict. No new target/geometry rows, model selection or diagnostic validation score.

Fresh physical GPU1 admission under /tmp/mto_pouter_gpu_1.lock must verify UUID, idle compute, required ECC schema/current mode/zero uncorrectable counters and no repair/recovery flags. Prebind UUID before child imports and register owned PID/start/boot/UID/cwd/argv plus log/status/terminal paths. No fallback or unrelated process action. Preserve actual failures and source bytes before diagnosis; no automatic retry. Any genuine blocker returns to root/reviewer without changing scientific settings.

## Freeze and production boundary

Before the GPU fixture, history binds objective/runner/reader/preflight/resources/wrapper plus all real imported dependencies and root decision. After it, history reviews actual receipts and source continuity. Final freeze includes exact source/protocol/settings, inherited proof mapping, CPU/GPU/negative authorization results, stats/split pins and all60 order hashes. Production commands must deny before data/model imports or artifacts without a separate exact root execution authorization, matching full closure, review and verified D-first publication receipt. No new TEST access, full fit, extension or seed allocation is implied by preparation success.
