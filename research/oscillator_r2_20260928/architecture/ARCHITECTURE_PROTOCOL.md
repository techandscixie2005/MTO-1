# Architecture screen: scalar oscillator strength and independent transition magnitude

Date: 2026-09-28. User permits architecture changes and use of all USTC-A800 GPUs; observed hardware faults still govern safe allocation. Scope: four controlled arms, not an open-ended automatic grid. No test evaluation is authorized by this protocol. Existing loss pilot is unchanged.

## Global state and hypothesis
Single-model benchmark remains eta0 epoch33: validation pooled raw-native-f R2=.4052941183 and historical test R2=.45581511. A separately frozen three-model eta ensemble has historical test R2=.48510176, delta+.02928666 (5.38% SSE reduction); molecule-bootstrap95% interval [+ .0047045,+ .0559655], positive fraction .99. This is an ensemble improvement on an already exposed historical test, not new single-model or untouched-holdout evidence. Architecture selection uses validation only; historical test outcomes do not choose losses, states or tail weights.

Channels64 G1 best available native-f validation R2=.40525454, essentially equal to eta0; simply widening the MTO has not supplied a clear gain. The original trace is beta^2+||Q_C||^2 and is multiplied by predicted E. We test whether a simpler independent scalar output improves conditioning and target alignment on the same learned molecular/state representation.

## Four arms, common objective
All use the eta0 core, MTO routing/query assembly, CG coupling, invariant readout hidden trunk and E head. All start from identical pretrained shared weights; residual additionally starts with exactly identical f. Hidden h is already conditioned on the molecular state; no new state embeddings or backbone blocks are added.

| Arm | Intensity prediction | Shape / parameter treatment | Hypothesis |
|---|---|---|---|
| original | f=c E tr(CC^T), original C readout | Original beta and tensor gates remain trainable | Matched objective/compute control |
| retained_residual | f=abs(f_base+f_std_train*delta(h)) | Original complete intensity plus4161parameter zero-output scalar MLP | Correct intensity while preserving tensor information and exact initialization |
| direct_f | f=f_std_train softplus(MLP_f(h)) | New4161parameter128-to32-to1 SiLU head; unused beta/tensor-gate heads removed | Direct scalar f avoids tensor-magnitude and energy-product constraints |
| independent_trace | s=s_std_train softplus(MLP_s(h)), f=c E s | Same4161parameter scalar MLP head; old shape heads frozen; normalized PSD shape exported | Independent scalar magnitude versus native-f parameterization/energy coupling |

For independent_trace, S=(A_old+epsilon I/3)/(tr(A_old)+epsilon), epsilon=1e-8; A=s*S. S is PSD with trace1 and s is independent of S. f is computed directly as c E s, with a preflight check against c E tr(A). Shape has no training penalty in this round. Shared hidden features still change with intensity training. This round cannot prove that normalized-shape supervision improves intensity, and no duplicate shape-only training arm is justified yet.

Common loss: LE=mean_valid((E_pred-E_true)^2)/sE2; Lf=mean_valid((f_pred-f_raw)^2)/Var_train(f_raw); L=LE+Lf. Variance is pooled over all valid training molecule/state entries, population convention, frozen once. It is neither per-state nor per-batch and never uses validation. LE uses the unchanged historical sE2=.5378066634062587. Raw printed f is primary; derived-f truth is a secondary diagnostic. Full predicted-E gradient is retained in original, retained_residual and independent_trace; direct_f decouples f output from E, while E supervision still trains the shared representation. No trace, Q, shape or spectral loss.

This loss differs from the earlier continuation pilot denominator; original control in THIS round is essential. Never compare a scalar-head improvement directly to that old pilot as pure architecture effect.

## Initialization and exposure accounting
Verified source eta0 checkpoint SHA256: 9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136. Frozen split:120355train/6686validation/6686historicaltest. prepare.py verifies source data hashes and draws exactly8192 training molecules by NumPy Generator(20260928). It saves IDs and indices; no validation/test predictions are read for initialization.

The source teacher is frozen. Cache its hidden h, E and native f on the subset. Initialize new scalar MLPs with zero final-layer weights and positive constant mean output; then fit each head for exactly1000 Adam AMSGrad steps, lr=.01, statebatch4096, seed20260928, clipping5, to the SAME teacher native-f MSE/Var_train(f), using identical state-index draws. Trace-head fitting multiplies its scalar trace by teacher E before computing this common target. Trunk/E/shape parameters receive no warmup updates. Save fitting errors, batch-order hash and timing separately.

Original and retained_residual reproduce their own teacher and need no head fitting. It receives the same training subset exposure record and later the same full-training updates; we do not waste compute optimizing an identically zero self-distillation loss. New-head warmup compute is additional and explicitly reported, never hidden as equal total compute. Distillation matches functions only approximately. Record each arm's epoch-zero validation and retain it in selection; poor initialization is distinguishable from later learning. No validation-based choice of warmup iteration or head scale.

Model parameter counts verified in PREPARATION.json: original total/trainable1552092; backbone1371840. direct_f total/trainable1551996. independent_trace total1556253,trainable1551996 because4257oldshape parameters are frozen. retained_residual total/trainable1556253, adding4161to original. New scalar heads are matched to each other and only96parameters smaller than original trainable count; residual is explicitly additional capacity.

The first129parameter linear-head diagnostic fitted teacher intensity poorly: normalized teacher MSE .18155/.18236. On train-only evidence, both new heads were replaced ONCE by128-to32-to1 SiLU MLPs,4161parameters, with the same1000steps and state order. Teacher errors fell to .06344/.06446. Linear artifacts remain under linear_diagnostic/ and are not completed architecture-performance experiments. No validation/test score selected head width.

Input audit: h carries M0/Ma per-channel invariants and CG scalars, compressed by the existing trunk. Original intensity additionally uses Gram products among32 equivariant tensor channels through ||sum_k g_k(h)T_k||^2. Those pairwise tensor invariants are not fully supplied to h. Direct-f and independent-trace heads therefore remove information as well as changing conditioning; they are not pure parameterization ablations. Warmup error quantifies approximation loss, not its unique cause.

Retained residual resolves that confound: delta(h) is a width32MLP whose final weight AND bias are zero. f=abs(f_base+f_std*delta) exactly preserves initial predictions and keeps the original tensor path trainable. For nonzero signed output, its delta derivative is +/-f_std, even for weak transitions; no clipped dead region. At exactly zero there is a cusp with PyTorch zero subgradient, and positive/negative latent values can represent the same f. Record exact-zero counts and signed-output quantiles. No residual distillation is needed. Primary f is computed directly. Auxiliary A is reconstructed from f/(cE) times normalized PSD shape only where E is finite and positive; tensor_reconstruction_valid flags other entries and they do not alter the primary f metric.


## Joint-training budget and selection
Fresh optimizer in every arm: Adam AMSGrad, lr1e-4, batch64, weight decay0, clipping5, FP32, AMP/TF32off, seed11 and orderseed11, two CPU threads, exactly20epochs, fixed lr, no scheduler. Only numeric/hardware faults stop early. Arm-specific distillation optimizer state is discarded. Shared full-training batch orders must have identical hashes. Same frozen data/masks and evaluation cadence.

Every epoch including0: native raw-f pooled float64 SSE/SST/R2/RMSE/MAE, per-state errors, fixed validation q90/q99 bright SSE, derived-f sensitivity, E MAE/R2 and common objective. Save best_native_f and best_common_objective independently; earliest tie wins. Original and retained_residual epoch-zero metrics must reproduce eta0 within the accepted inference tolerance. New-head epoch-zero metrics need not equal eta0 and must be reported explicitly.

Root may extend ALL arms by10matched epochs only if a new-head arm's best is at epoch18or later, its best R2 in epochs16-20 exceeds its best R2 in epochs11-15 by>=.005, and it is within.01R2 of matched control or better. This is a prespecified late-learning diagnostic, not automatic queue behavior; root reviews cost and evidence first. Otherwise20epochs is final for this screen.

Promotion: delta validation native-f R2>=.01 versus THIS round's control and improvement in>=80% of2000 molecule-paired bootstrap resamples; .005-.01 is tentative. E MAE>110% of own epochzero is a review flag, not an automatic veto. The user's f R2 objective remains primary. Bootstrap after validation checkpoint selection is descriptive and does not remove selection optimism. Do not delete hard examples, select states, reweight tails or choose an architecture using exposed test results.

Root freezes the selected candidate before any later historical-test comparison. A promising architecture needs matched-seed and/or from-scratch confirmation, and genuinely untouched data for strong generalization claims. No large architecture search is launched merely because GPUs exist.

## Implementation, review and hardware
All additive files: research/oscillator_r2_20260928/architecture/. model.py exposes build(arm,stats,load_initial=True); forward returns dict E,f,A,trace. direct_f has no predicted A/trace. objective.py exposes terms(pred,target,stats) and native_f64(pred,arm). prepare.py writes stats.json, warmup_subset.json, initial/*.pt, cache/teacher_features.pt and PREPARATION.json. The executor supplies separate train/preflight/scheduler/summary and reviews these modules before launch. Never mutate the approved loss-pilot files or source experiment.

Required smoke checks: original source-forward equivalence, finite gradients for each arm, same shared pretrained weights, direct_f f independent of E head, trace f has expected E gradient, PSD/unit-trace shape and f/trace consistency, rotation/permutation behavior on source examples, valid masks/raw indexing, save/resume equivalence, epoch0 selection, source/data/init hashes. Parameter counts and train-only variance must be recorded.

GPU5 is permitted for guarded short runs: one historical corrected DRAM error, no uncorrectable/remap/pending, numerical probe passed, require unchanged ECC counters. Clean released GPUs1/4/6 may run arms concurrently after health/occupancy/lock checks. Prioritize original control and retained_residual, then direct_f and independent_trace; allocation is dynamic across admitted GPUs5/1/4/6 rather than waiting for a particular release. GPU2 remains assigned to loss pilot. GPUs3/7 are excluded due observed uncorrectable ECC and pending remaps. GPU0's unrelated SpecGPT is preserved. No resets or interference with existing supervisor jobs. Dedicated monitoring every4h remains required; operational health watchdogs may act sooner.

Observed older training throughput approximately145-150sec/epoch suggests about50minutes/GPU per20epoch arm, plus initialization and validation. Record actual times; estimates are not guarantees. Archive code/config/lightweight reports to local D: first, then GitHub. Never upload checkpoints, feature cache or raw tensor arrays.
