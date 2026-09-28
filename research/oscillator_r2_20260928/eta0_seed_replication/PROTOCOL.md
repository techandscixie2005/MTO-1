# Eta0 seed23/37: bounded 100-epoch replication

Date: 2026-09-29. Root accepted preparation of exactly two fresh runs, seeds23 and37. Execution requires independent implementation/preflight PASS and root launch authorization. The approved frozen Gram study remains unchanged. No test access, new split, seed replacement, weight/subset search, or extra epochs.

## Question and fixed comparisons

Does fresh-seed variation preserve eta0 quality and provide useful complementary errors at the same three-model inference cost as the existing three-loss-setting ensemble? This tests initialization/order robustness and averaging, not a corrected-objective or architecture hypothesis. Seeds23/37 are fixed before outcomes; no provenance-based reason to replace them was found.

Primary checkpoints: each new run's minimum LEGACY validation joint objective among epochs1..100, earliest exact tie. The historical seed11 checkpoint remains its archived legacy-selected epoch33. Its global best over236epochs was33, so it is also the minimum among its first100 legacy evaluations. The new runs reproduce the first100-epoch policy, not the historical236-epoch early-stop duration.

Primary ensemble: arithmetic mean of native raw-f predictions from these three legacy-selected seeds11/23/37, equal weights1/3. Compare once with archived eta0 and the already fixed equal-three eta0/eta0.1/eta1 ensemble. Report each seed individually. No selection of the better seed, two-seed subsets, or fit of ensemble weights.

Also save each new run's independently minimum pooled raw-f validation SSE among epochs1..100, earliest exact tie; label these SECONDARY raw-f-selected checkpoints. Report their individual metrics and a prespecified equal-three MIXED-SELECTION ensemble (old seed11 legacy-selected plus new23/37 raw-f-selected). It is not a uniformly raw-f-selected three-seed ensemble: old per-epoch checkpoints are unavailable. Never substitute secondary checkpoints into the primary ensemble or declare seed11 raw-f-optimal. Evaluate/save epoch0 for diagnostics only; it is not a selection candidate because the legacy run did not select epoch0.

## Exact legacy training behavior

Use immutable source experiments/qm9s_eta_Ef_20260926/configs/mto_eta0.json, model_factory.py, frozen_reference model/core, objective.py, normalization and split. Do not edit these files. New code/configs live in this study directory. Only seed/name/output/device and max_epochs=100 change the scientific training configuration. Preserve min_epochs200 and legacy early-stop logic (therefore no early stop within100); do not feed new native-f diagnostics to the scheduler.

Architecture: original MTOEA, 10states, MTO channels16, query32, router/head128, same complete DetaNet backbone, total1,552,092/backbone1,371,840 parameters. Fresh random initialization through legacy build(cfg,stats); train-only E_state_mean energy offsets and n_ref18 remain unchanged. No source-model/best/residual checkpoint warm start and no head replacement or distillation.

Objective eta=0:

    L = mean_mask_E[(Epred-Etarget)^2]/sE2
        + mean_mask_A[(tr(Apred)-tr(Atarget))^2]/(3*sA2)
    sE2=.5378066634062587; sA2=.1324285377060015

This is the legacy trace objective, not native-f variance normalization; do not substitute the architecture-screen loss or claim the seed study resolves that confound. Retain the legacy tensor diagnostic term calculation with coefficient0. Use unchanged target E/A, raw f only for diagnostics/secondary selection. Never supply true E to f prediction.

Adam AMSGrad, lr.001, betas(.9,.999), eps1e-8, weight_decay0, clip global gradient norm5 with nonfinite failure, FP32, AMP/TF32 off, CPU threads2. ReduceLROnPlateau on legacy validation L after each epoch: factor.5, patience50, relative threshold1e-4, min_lr1e-6, other defaults matching pinned runtime. Epoch best comparison precedes scheduler step, exactly as legacy. Save actual LR each epoch and full scheduler state.

Fixed connectivity split120355train/6686val/6686untouchedtest. Batch64molecules, all10states/masks, 1881updates per epoch including final35, exactly188100updates/run. NumPy default_rng(seed).permutation(original_train_indices) once per epoch. No reshuffling from evaluation or different data-order generator. Save all100 order SHA256 values and explicit initialization/seed/source provenance. Different seeds intentionally change both initialization and order, so this is not a pure initialization-only experiment.

## Evaluation and diagnostics

Every epoch: legacy validation objective in original FP32 path/aggregation; independent raw-f pooled FP64 SSE/SST/R2/MAE/RMSE using raw printed f and evaluator-compatible native prediction (promote original FP32 E/A outputs to FP64 before trace/native conversion). Preserve original PSD-derived nonnegative f with no extra abs/clip/calibration. Verify exact IDs/global indices/state slots/masks and raw labels. E MAE, per-state f SSE and fixed q90=.0546/q99=.2377 tails/complements are diagnostics. Avoid unnecessary spectrum work only if demonstrated to have no RNG/model/selection effect; scientific legacy objective and scheduler remain exact.

Save initial, legacy-best, raw-f-best and final100 model checkpoints and validation prediction arrays server-only; aliases may point to the same checkpoint when selections coincide. Save complete history including selection transitions, losses, learning rates, unclipped gradient norm summaries/clipping counts, timing, steps and order hashes. After fit, evaluate full training metrics at initial, both selected checkpoints and final to distinguish training fit from generalization. Online changing-model training averages are not full checkpoint-fixed train SSE. No validation-driven extension or retraining.

At completion evaluate the prespecified comparisons with2000 paired whole-molecule and connectivity-group bootstrap replicates, seed20260928, all10states together and pooled SST recomputed. Report residual-error correlations, tail/state gains, molecule concentration and E tradeoffs. Three seeds provide a small robustness sample, not a precise population seed variance. Checkpoint selection and extensive prior validation reuse make confidence intervals descriptive. A practical validation candidate needs >=.01 pooled raw-f R2 gain over eta0 and positive delta in >=80% of molecule draws, with group robustness and equal-cost equal-three comparison reported; no automatic promotion or test authorization.

## Preflight and immutable provenance

Implementation handoff must pin all actually executed sources, configuration, protocol, dataset/split/scalers, raw-label source and fixed comparison arrays. A new train/val-only adapter may replace legacy eager loading without altering tensors, masks, graphs or order. Verify on fixed training rows against the original batch/objective path. Array storage decoding is not itself evidence of training leakage; no test batches, test label consumption, or test metrics are allowed.

For each seed: reproduce fresh initialization twice with identical state hashes, record distinct hashes across23/37 and relative to archived seed11 initialization. Check named random core weights differ even if deterministic buffers/output offsets match. Verify parameter counts, energy offsets, no checkpoint-load path, and initial finite native metrics. A seed11 construction may be compared to archived initialization hash as a source-compatibility check; do not run a third training study.

Verify deterministic100order hashes before launch, each order exactly the same training index set with no omissions/duplicates, seeds differ,1881updates and last35. Test original versus adapted batch/loss/gradients/one-step update on a bounded fixed training batch with identical copied model states; test a two-step interrupted/resumed path reproduces uninterrupted model/optimizer/scheduler/RNG/order/cursor. Ensure diagnostics do not consume training RNG or change train/eval mode before updates. Resume requires exact source/config/seed hashes and stores CPU/CUDA/Python/NumPy global plus order-generator RNG, optimizer, scheduler, epoch/cursor/order, sums/counts and both selection states; atomic checkpoints at least every600s and epoch end. Never silently restart or overwrite a failed run.

Use per-run lock plus shared physical-GPU lock, healthy-idle admission, UUID/ECC/remap/occupancy receipts and continuing epoch/checkpoint health checks. Target GPUs1 and6 AFTER the short Gram passes release their lock. Preserve unrelatedGPU0 and other jobs; exclude faulty3/7 and do not move to guarded5 without its separate admission. On health drift/nonfinite/provenance mismatch, stop/checkpoint and mark invalid or awaiting review; do not preempt/reset hardware.

## Cost and handoff

Original eta0 took35,296.3s for236epochs (~149.6s/epoch). Budget about4.15GPU-hours per100epoch run,8.3GPU-hours total before additional diagnostics and IO; ~4.2h wall if two admitted devices run concurrently. Record actual utilization/time rather than assuming a gain. Parallel seed runs are justified by the independent scientific question; they do not wait for a favorable Gram result, only released resources and review gates.

Implementation owner: remote_audit. Reviewer: backbone_baseline_audit. Required launch artifacts: protocol/config/source manifests, initialization/order provenance, bounded preflight PASS, exact-hash independent implementation PASS and root authorization. This file is a protocol/handoff, not permission to bypass those gates.