# Frozen eta0 residual head: fixed protocol

Date: 2026-09-29. Scope: ONE run, seed11, twenty epochs, train/validation only. Root standing authorization covers preparation and launch only after completed architecture integrity review, reviewed final20 residual replay, implementation review PASS, preflight PASS and healthy idle GPU/shared-lock admission. No user reconfirmation is required. This protocol itself runs nothing. Never modify the sealed architecture/loss studies.

## Question and comparison

Does the existing frozen eta0 representation support a useful additive correction when the molecular backbone, routing, tensor base and energy predictions cannot drift? Compare a single frozen-base head fit with the unchanged eta0 predictor (epoch0 control) and the completed joint retained_residual run. Freezing is the intended intervention; cache computation is only an implementation optimization. The older joint run is not repeated. Equal training examples/updates and head initialization are required; lower compute for the frozen run is intentional and must be reported.

Current benchmark is pooled raw validation f R2=.4052941183. All comparisons are exploratory on the already-used validation split. No test labels/predictions or historical-test comparison is authorized. No from-scratch backbone/readout grid, new width, optimizer sweep, training extension, residual clipping or state/tail reweighting is included.

## Fixed model and initialization

Use source eta0 epoch33 checkpoint SHA2569f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136 and sealed architecture/initial/retained_residual.pt SHA256ae29b2a7c657fec8566e3e230b55a70c385377b36dd37e08c0dcca873f3c7a8d. Copy the EXACT width32 residual head state (128-to32-to1, SiLU), including its existing random first layer and zero final affine weight/bias. Do not redraw the first layer, warm up, distill, or start from the trained final20 head.

Only the4161 residual-head parameters require gradients and enter the optimizer. All1,552,092 source-model parameters, all source buffers, hidden h, predicted E and original tensor-derived base outputs remain frozen. Use eval mode for feature generation. Seed11 is retained for reproducibility, but copying the sealed initial head is authoritative for initial parameter identity.

Prediction retains the same parameterization:

    delta32 = f_std_train * MLP(h32)
    train_f32 = abs(base_f32 + delta32)
    eval_f64 = abs(base_f_native64 + delta32.double())

f_std_train=.050109692115345626. Frozen base_f32 is the existing model float32 path C_F*E32*trace(base_A32); base_f_native64 is C_F*E32.double()*trace(base_A32.double()), with the diagonal promoted before its sum and multiplication. C_F=2/(3*27.211386245988). Use these separate paths exactly as in architecture/model.py and objective.py; do not obtain native64 by simply casting base_f32. Energy is predicted energy only. Raw true E/f never enter h, base or model inputs.

Initial delta is exactly zero, so epoch0 reproduces the source predictions within the established native-f tolerance. The zero last layer allows its own gradients immediately; hidden residual-layer gradients begin after last-layer weights move. Abs has a cusp at signed0; record exact-zero and negative-signed counts rather than adding a new activation policy.

## Frozen cache and data alignment

Frozen split:120355 training and6686 validation molecules, ten ordered states each. Same original raw printed f, masks, graph cache, IDs/indices and input z/pos. No test partition is consumed. Validation truth must use the frozen raw validation export with verified provenance; training labels must be indexed only by the fixed training split. Bytewise source-file hashing is integrity checking, not target use. Do not instantiate the old all-split Data loader for this new diagnostic.

Generate one additive server-only cache per train/validation split with FP32 h[N,10,128], E32, base_f32, native64 base_f, raw f truth/mask, IDs and global indices. Cache energy target/mask for reporting only. Record source checkpoint, initial head, data/split/input graph, stats, feature generator and cache file hashes. Include exact dtype/shape/schema, split order and model-eval settings. Do not use validation targets during feature computation or choose cache precision from validation performance.

FP32 hidden storage is about616.2MB train and34.2MB validation (decimal), plus small scalar/ID arrays. Cache and optional device copies avoid repeating expensive model forwards each epoch. Use bounded batch64 generation on a healthy admitted GPU with two CPU threads; record actual time/memory. No throughput promise is assumed. All caches, raw arrays and checkpoints remain on USTC-A800; only lightweight code/config/reviews/metrics may go to D:/GitHub.

## Loss and exact update budget

Train Lf=mean_valid((train_f32-f_raw32)^2)/.002510981243894732, population pooled train f variance. For comparable reporting retain LE=mean_valid((E_frozen-E_true)^2)/.5378066634062587 and LE+Lf. LE is constant with respect to all trainable parameters, so optimizing Lf alone gives the same head gradient as LE+Lf. Do not rescale the head objective, use per-state variances or refit normalization. The prior lower-weight pilot remains a different objective comparison.

One fresh Adam AMSGrad: lr1e-4, betas(.9,.999), eps1e-8, weight_decay0. Clip global norm of the trainable head to5. Batch64 MOLECULES with all ten states together (not randomly flattened states). FP32 optimization, AMP/TF32off, two CPU threads. Seed/orderseed11; exact same NumPy Generator permutation sequence and molecule order hashes as the completed residual arm. Each epoch has1881 updates including the final35-molecule batch;20epochs gives37620 updates. No scheduler, warmup, early stopping or extra epochs. Numerical/hardware/integrity failure stops with an explicit incomplete/invalid receipt.

Freezing intentionally changes the gradient set and clipping norm relative to full-model training; log unclipped head gradient norm and clipping frequency so this is visible. It does not change the nominal optimizer policy. Record cache, head-training and evaluation wall times separately.

## Evaluation, selection and retained diagnostics

Evaluate epoch0 and every completed epoch using the entire fixed validation split and native64 f. Select minimum raw pooled validation f SSE, earliest exact tie, with epoch0 eligible. No alternate state/MAE/energy/tail metric can select the primary checkpoint. Maintain last checkpoint and best checkpoint with optimizer/RNG/order/cursor state sufficient for exact resume. Save separate server-only initial, best and final validation predictions; final20 must be saved even when best remains0.

Every epoch also evaluate full frozen-feature TRAIN data at the current fixed head checkpoint. Report raw train f SSE/R2/RMSE/MAE separately from online optimization loss, with its own train SST; report both train/val common objective components. This removes the previous combined-online-loss ambiguity. Keep the train truth at raw source precision for native evaluation and float32 for training, matching source conventions.

Validation reporting: pooled float64 raw-f SSE/SST/R2/MAE/RMSE, per-state metrics, fixed true-f q90=.0546/q99=.2377 and complementary SSE, E metrics, signed/base/delta quantiles and extrema, negative/zero signed counts, residual magnitude, and direct base-versus-corrected SSE decomposition. Cache/frozen E/base hashes or exact tensor comparisons must demonstrate invariance through training. Use the same frozen baseline in all diagnostic comparisons. Save ID/index/raw truth/base32/base64/delta/signed/final/E columns in server-only best and final arrays; no tensor A export is required for this scalar diagnostic.

After selection, compute paired molecule and connectivity-group bootstrap deltas versus frozen eta0 using2000 replicates, seed20260928, preserving all ten states/molecule and recomputing pooled SST. Report final20 and the completed joint residual's selected/final performance separately; do not confuse matched fixed-epoch behavior with selected-model superiority. Group layout and target alignment must reproduce the completed addendum.

## Preflight and independent-review gates

Before training the executor supplies sealed manifest/config, cache/preflight and independent code review. Required checks:

1. Verify source/init/data/split/stats hashes and exact copied initial head. Assert only4161 head parameters trainable and optimizer membership exactly matches them.
2. Prove cache identity on a fixed training batch and fixed validation batch against the uncached frozen-source forward, comparing h/E/base32/base64 and zero-initial/final f. Verify IDs, state order, masks and edge offsets. Check finite values and exact epoch0 delta0; reproduce validation eta0 R2 within5e-7 (tighten if existing preflight already requires tighter). Fail rather than tolerate material inference drift.
3. Compare cached versus uncached frozen-model head loss, head gradients and a copied one-step Adam update on a fixed training batch, allowing only declared FP32 numerical tolerance (default abs1e-6/rel1e-5 for nonzero tensors). No backbone gradient, parameter update or cache mutation may occur. Retain existing invariance tests on source examples; cache equivalence ensures their prediction semantics are preserved.
4. Check train/validation disjointness and all batch IDs; reproduce all20 archived residual order hashes before launch. Verify1881 updates/epoch and final batch35. No test split inspection is required for these checks.
5. Verify resumed versus uninterrupted short training with exact RNG/order/cursor/optimizer restoration; best/epoch0 selection, final prediction save and atomic writes. Preflight uses disposable additive outputs, never historical checkpoints.
6. Require completed architecture integrity PASS, final20 replay receipt, and this run's independent implementation review PASS. Admit only an idle eligible GPU1/4/5/6 under /tmp/mto_pouter_gpu_<gpu>.lock. GPU5 additionally follows pinned ECC/microcheck policy. Preserve GPU0, occupied GPU2, and excluded faulty3/7; no preemption/reset. Check health before/after cache and training and through existing bounded watchdogs. No new monitoring cadence is created here.

## Decision criteria

A selected gain>=.01 raw validation R2 and positive delta in>=80% of paired molecule bootstrap draws is enough to nominate this frozen correction for later validation confirmation; require connectivity-bootstrap reporting as a robustness check. Gains.005-.01 are tentative. This remains one-seed, validation-selected exploratory evidence, not a confirmed generalization claim or permission to inspect test. Numerical near-ties never count as gains. Frozen E cannot trade accuracy for f, and a changing E metric is an integrity failure rather than a model tradeoff.

If training SSE improves while validation never beats epoch0, conclude that this frozen-feature head fit did not yield a useful correction under the fixed settings; do not infer an intrinsic MTO/backbone limit. If it improves versus frozen eta0 while joint training failed, preserving the representation becomes the preferred explanation to investigate; freezing also changes gradient/clipping dynamics, so do not claim a uniquely proven causal mechanism. No failed outcome automatically authorizes a learning-rate search, longer run, larger head or100epoch scratch pair.