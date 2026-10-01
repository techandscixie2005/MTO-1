# Round06: fresh original-PSD objective comparison

## Question and authority

Does directly supervising native printed oscillator strength improve validation accuracy when the original PSD readout, initialization, data order, optimizer and budget are held fixed? This is an objective comparison. It does not test a new quantum-chemical operator or establish why previous fits failed.

The accepted scientific specification remains `PROTOCOL_PROPOSAL.md` (SHA256 `aea5c76f422fb7cca286cd84f7bafaff230e2aa29b2ed3068fe7e4573f3245b6`) with independent proposal PASS `a1617eaea6574b36f5e39fa9b5bb738c7a2ba91083c373217e2c51d25b992225`. This document describes the implementation, completed engineering scope and future execution boundary. `ROUND06_PREPARATION_DECISION.md` authorizes preparation only. Publication, a preparation PASS or a heartbeat does not itself authorize production.

## Matched arms and loss

| Arm | Objective | Architecture |
|---|---|---|
| trace_control | LE + Ls | Original PSD MTO |
| raw_f | LE + Lf | Identical original PSD MTO |

The wrapper retains the same checkpoint schema in both arms. Its unused right-F parameters are frozen and the adapter is disabled, so they do not change the original function. All original backbone, PSD-head and energy-offset parameters train. There is no decorrelation penalty, Q supervision, auxiliary loss, calibration or new QC label dependency.

Let E be predicted energy in eV, A the predicted PSD dipole outer-product matrix in atomic units, and s=tr(A). Both training and evaluation use native f = c E s, with c = 2/(3×27.211386245988). No target-energy substitution, detached branch, absolute value, clipping, cap or logarithm is introduced. The unchanged model uses softplus energy and PSD A, giving its existing nonnegative native-f contract; LE also remains unchanged. The inherited evaluator rejects negative or nonfinite native-f predictions rather than repairing them; such a failure must be preserved and diagnosed.

- LE is the original masked mean squared energy error divided by TRAIN sE2=.5376833706691944.
- Ls is the original squared trace error averaged over mask_E & mask_A, divided by 3×sA2, with sA2=.13245475393297818. The original `base_loss` code and its linear trace extraction are retained.
- Lf is mean squared error against the printed raw-f labels, averaged over mask_E & mask_f, divided by the new TRAIN population variance .0025089892829484074. This is pooled population variance, including valid zeros, not a per-state variance or reconstructed-label statistic.
- The two terms have fixed coefficient1. Dimensionless normalization does not equalize their gradients. No balancing, learning-rate fallback or validation-driven coefficient change is allowed.

For Lf, valid entries are selected before E×tr(A), subtraction and squared error. Invalid sentinel entries cannot contaminate gradients. The production dataset contract requires every frozen E/A/f mask to remain true and rejects changed validity rather than excluding rows. All 1,203,550 TRAIN labels, including22,585 printed zeros, are included. Model outputs and training f targets are FP32. Evaluation retains FP64 target f and computes native f from FP64-converted E/A. The tiny resulting precision distinction is deliberate and reported.

The product loss sends gradients through both predicted E and A. For residual r=cEs−f, dLf/dE=2csr/(N Var(f)), dLf/dA=[2cEr/(N Var(f))]I. If A=CCᵀ, dLf/dC=4cErC/(N Var(f)); at C=0, factor gradients can vanish even when the direct A gradient is nonzero. Energy/intensity compensation and sensitivity to rare large errors remain empirical risks. The existing PSD tensor directions are learned features; no electronic phase, transition-density or polarization identification follows from this loss.

## Partition and target boundary

Use only the sealed v2 split: manifest `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`, independent verification `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`. Counts are120,355 TRAIN,6,686 validation and6,686 TEST. Statistics are the exact previously audited TRAIN-only bytes, SHA `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.

`partition_data.py` is byte-identical to the reviewed Round05 reader. Membership is checked before opening a ZIP member. Only selected contiguous row blocks are converted to numeric values; actual fields, index hashes and row counts are recorded. Reading/compressing/hashing opaque archive bytes does not imply numeric target decoding. There is no TEST reader API. Validation construction requires the production caller to pass the exact authorization gate first; it is not part of technical preparation.

This is an internally repartitioned, historically exposed corpus, not a new external holdout. New TEST includes5,989 old-TRAIN,361 old-validation and336 old-TEST rows. Therefore both production models must be freshly initialized; all historical trained weights are forbidden. Numeric TEST targets and all TEST predictions remain inaccessible in this round.

## Common training and selection

Fresh original MTO16, query32, router/head128. Rebuild seed11 random initialization independently of all disposable technical states. Common original-base hash is `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; full-schema hash is `3c5d20463a6aa1d4e7c25ee5e57e0ceaa53c17e5a89af35d7761b58e58d661f9`. Both are checked before fitting. Model construction and data-order RNG are isolated; NumPy PCG64 order seed11 generates identical60 orders, frozen in the manifest.

Adam AMSGrad, lr=.001 fixed, betas(.9,.999), eps1e-8, weight decay0, batch64, gradient clip5, FP32, AMP/TF32 off, two CPU threads, no scheduler. Each arm completes exactly60 epochs,1,881 updates per epoch and112,860 updates total unless an actual failure interrupts it. No early stopping or extension. No preflight parameters or optimizer states may initialize production. Identical tensors/orders do not imply bitwise deterministic CUDA trajectories.

Select the earliest minimum pooled validation raw-f SSE from epochs0–60, using all66,860 valid labels. With fixed truth/count, this maximizes pooled R². Report both independently selected checkpoints and fixed epoch60; do not average predictions. Report per-state SSE/MAE/RMSE/R²/counts, energy error, true bright-tail errors and all four true/predicted-bright bins. TRAIN-only thresholds are q90=.0549 and q99=.2406. No exclusions or state permutations can improve the score. Any later paired component bootstrap is descriptive uncertainty conditional on selected validation checkpoints, not independent-seed or selection uncertainty.

Candidate allocation requires selected raw_f R² at least .003 above BOTH its contemporaneous selected trace_control and the retained Round05 v2 reference .44716940136585204. The exact threshold against that reference is .45016940136585204. Report energy/state/tail tradeoffs even if the pooled gate passes. Beating only a weaker rerun is insufficient. A control improvement is descriptive execution variation, not evidence for Lf. Neither outcome authorizes an extension or new seeds automatically; root must decide after completed results are independently reviewed and archived. A later confirmation, if authorized, pairs genuinely fresh seeds23/37 without prediction averaging.

## Log interpretation and recovery

TRAIN `total` is the selected arm objective; TRAIN `base` and validation `base_objective` remain LE+Ls in both arms. TRAIN `normalized_raw_f` is FP32 Lf; `raw_f_mse` is the separate FP64 native-f diagnostic before variance division. TRAIN epoch losses/gradients/features are weighted aggregates of pre-update minibatches along the trajectory, not a fixed final-checkpoint TRAIN evaluation. Validation is evaluated at each completed checkpoint. The dormant-F movement must remain zero; raw-M decorrelation is logged without gradient or loss contribution.

Each completed epoch atomically commits `last.pt` with model, Adam state, RNG/order state, config/statistics/manifest bindings, step count, history and selected prediction/checkpoint references. `last.pt` is the committed restart point. An interruption mid-epoch replays that entire epoch from the preceding committed state. `best.pt` retains the selected model/optimizer/RNG snapshot, but its internal selection metadata precedes its own checksum; it is not the runner's committed resume entry. Immutable selected versions prevent an interrupted best/last copy from silently changing selection. File and parent-directory fsync are inherited unchanged.

Checkpoint/source/identity mismatches stop. Existing failed markers are preserved on a successful explicit resume. A live same-arm worker lock or completed FIT_COMPLETE blocks relaunch. Recovery requires actual process/receipt diagnosis, not a blind retry. All checkpoint and prediction arrays remain server-only.

## Completed bounded engineering evidence

CPU synthetic checks verify exact original control loss, independent E/A/C analytical gradients, valid zeros, invalid-NaN backward masking, empty/invalid contract rejection, O(3) objective invariance and identical fresh initialization. Existing model O(3)/permutation/translation and reader/constructor checks are inherited through the exact Round05 source closure. The independent production-gate suite has22 total authorization checks, including a valid synthetic binding and rejection before scientific imports/artifacts.

The GPU fixture executed exactly6 discarded updates on the first128 ascending TRAIN rows: two batches64, then restoration/replay of the second update, per arm. Its12 actual decoding ledger entries each contain the same128 TRAIN rows; validation/TEST numeric rows are0. All six calls use the actual production objective. RNG/order replay is exact; model/Adam numerical tolerances are atol2e-6, rtol1e-5 and loss replay atol1e-6, rtol1e-5, fixed before execution. Full-GPU E/A identity/export uses the already documented numerical parity policy; bitwise equality is not claimed.

Actual model replay maximum difference1.1920928955078125e-7, Adam1.4901161193847656e-8, loss0. One-checkpoint construction and forward access guards passed on the same first two TRAIN geometries. At the initial fixed minibatch, raw-f intensity gradient norm was about6.7829 times trace-intensity gradient norm. Both branches were live and all six steps clipped at the frozen norm5. No coefficient, learning rate or tolerance changed. These are engineering observations, not accuracy evidence.

The wrapper finished in19.01s; three-update sections took2.79s and2.11s, with peak allocated memory below621MB. This small early-row fixture does not represent full-corpus maximum molecule/batch sizes. Historical full Round05 timing supports roughly2.7h wall time for two concurrent60-epoch arms, about5.4GPUh before overhead; reserve approximately25% timing margin without changing the budget.

The first prelaunch invocation rejected a missing mirrored ENVIRONMENT.json before stage creation/admission/child/data/update. The already-reviewed exact file was mirrored, all116 source pins independently rechecked, and the next explicit invocation was the only actual GPU fixture. `ops/GPU_PRELAUNCH_GUARD_FAILURE.md` preserves the event. No scientific source or acceptance threshold changed. Disposable fixture weights remain private and must never enter production.

## Future commands and execution gate

Server directory: `/home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation`.

The following production command is **blocked now**. A distinct root `PRODUCTION_EXECUTION_AUTHORIZATION.json` must first bind the final FROZEN_MANIFEST, exact independent preparation review, sealed split/verifier and remotely verified D-first publication receipt. The provided template has `authorized:false`. Preparation review/publication alone cannot release this gate.

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /usr/bin/python3 launch.py --authorization "$PWD/PRODUCTION_EXECUTION_AUTHORIZATION.json"
```

The reviewed launcher admits trace_control on physicalGPU1 first, then raw_f on physicalGPU2. Each has a separate exclusive shared GPU lock inherited by its child. Admission checks the exact UUID, idle status, ECC schema/counters and repair/recovery flags. Child CUDA visibility is bound to the physical UUID before Python imports (logical cuda:0 inside each child), with no fallback GPU. GPU1 UUID is `GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800`; GPU2 is `GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa`. The two admitted fits may run concurrently. Sequential admission does not promise an all-or-none launch: if the second admission fails, preserve and report the exact first-worker state; do not relaunch it. GPU0 and all unrelated processes remain untouched.

Owned PID/start ticks/boot/UID/cwd/argv hash, UUID/environment, command, attempt logs, resource XML and registration are persisted. Existing four-hour monitoring is retained. On any failure, preserve evidence and identify ownership before an explicit recovery decision. An authorized recovery can use `--arms trace_control` or `--arms raw_f` with the same bound authorization after verifying that arm is not live or completed; no settings change is allowed.

The terminal geometry export `runs/<arm>/geometry_best.pt` contains the full model/config/TRAIN statistics and reconstructed-buffer checks. `predictor.load_predictor(checkpoint, device)` loads it without reopening base checkpoints, datasets or caches, then takes only z/positions/geometry edges. Obtain f with the same c E tr(A) expression. The final predictor uses one such checkpoint. Restore the exact lightweight dependency paths from the manifest/archive mapping; weights are never part of the repository.

## Historical distinction and limits

The audited historical .373493 scratch study replaced the original beta/tensor PSD readout with a direct-f head, changed energy initialization/offset behavior and used100 epochs. Earlier original-PSD raw-f/E² objectives were trained from eta0 warmstarts at lr1e-4; frozen-F objectives kept the base fixed. Those negative results remain relevant but do not duplicate this matched fresh unchanged-PSD contrast. Round05 LE+Ls F/decor scratch arms also do not test this objective. This is a claim about the audited records, not proof no other experiment exists anywhere.

The .60 goal is unachieved. A single seed, fixed loss coefficient, reused validation and an internally repartitioned corpus limit conclusions. No experimental improvement follows from stable gradients, rank, replay or export parity. Physical state phase/degeneracy, full-TDDFT amplitudes and AO/MO conventions are not resolved by directly supervising scalar oscillator strength. No unavailable QC labels are required at inference.
