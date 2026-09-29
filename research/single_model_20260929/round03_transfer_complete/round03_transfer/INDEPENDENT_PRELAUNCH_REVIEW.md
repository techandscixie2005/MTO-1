# Round03 independent prelaunch review

**PASS for the exact fixed source-generation and affine-transfer stage, conditional on the archived preparation publication and immediate resource gates.** This approves no nonlinear readout fit, source checkpoint search or test evaluation.

Frozen manifest SHA: `93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37`. All 69 referenced files, including source data/index hashes, were rechecked against the current server contents. Source fitting had not started at review.

## Scientific and access checks

- The original outer split is preserved. The internal group split matches both approved index hashes and assigns every original TRAIN molecule exactly once: 96,284 fit and 24,071 calibration. Unresolved groups remain fit-only; labels do not determine assignment.
- Source normalization uses raw FP64 E/A from the fit subset only. All 962,840 E and A state labels remain valid. The fitting loader decodes only fitting geometry, E/A and their masks; no f, calibration, validation or test target enters fitting.
- Fresh original `MTOEA` initialization uses seed11 and the fit-only energy offsets. Guarded construction opened no checkpoint, dataset or normalization file. The preflight initial tensor hash is `700349ff3855c1d461b24d5b0bf9d2efa038ab143c1385e427e5b2ab9d679b90`.
- Source training has no held-out evaluator or best-checkpoint selection. It uses exactly 33 epochs of original LE+Ls with AMSGrad, lr0.001, batch64, clip5, weight decay0, FP32 and the original optimizer defaults. Original loss and gradients matched bitwise on the actual fit batch.
- The fixed source has 49,665 updates, versus 62,073 in the original full-TRAIN 33-epoch prefix. Training size and resulting source quality remain confounders; this is an internal transfer comparison, not isolated proof of a membership mechanism or fresh outer validation.
- Both maps fit the same calibration labels using one unconstrained FP64 pooled affine least-squares calculation each. All valid raw-f labels, including zeros, are retained. Invalid inputs or zero variance fail explicitly. The source is fixed epoch33 regardless of these outcomes.
- Both coefficient pairs are recorded and hashed before the outer-validation stage. Export recovery reuses the immutable coefficients and pinned prediction array; it does not refit. Each map applies directly to the same full baseline's native f. No historical calibration composition, temporary-source deployment dependency or prediction averaging occurs.
- The outer evaluator checks all 66,860 valid labels and replays native and historical-calibration R² within1e-7 of their recorded anchors. It reports state and fixed TRAIN-tail metrics. The allocation rule requires at least .003 R² above both the in-sample map and native anchor, with root assessment of tradeoffs. It permits only subsequent nonlinear-readout preparation.

## Checkpoints, recovery and inference

Atomic epoch checkpoints retain model, optimizer, source config, statistics, manifest, RNG and completed history. A partial epoch replays from the preceding complete checkpoint. Preflight verified exact data-order continuation and restored RNG state. The next model update differed by at most `2.384185791015625e-7`, and optimizer tensors by `1.1175870895385742e-8`, within the declared `atol2e-6/rtol1e-5`. This is numerical next-update parity, not a claim of bitwise GPU resume; its cause was not isolated by this check.

The CPU one-checkpoint fixture reproduced E, A, native f and affine f bitwise and accessed no original checkpoint, cache or data during inference. The final checkpoint embeds the full baseline, exact configuration/statistics and one coefficient pair. Geometry is the only inference input. Affine f can differ from the native E/trace(A) relation; E/A are unchanged outputs.

## Operational checks and remaining gates

The launcher binds the exact review, manifest and archive-first publication receipt. It requires completed round02 publication, GPU admission and shared locks, then registers the owned source process with the existing four-hour monitor. Runtime hashes are checked again before fitting and each affine stage. Full checkpoint/cache/data/prediction arrays remain server-only; only explicit lightweight records enter archiving.

Both requested robustness changes are present: interrupted exports reuse frozen coefficients, and outer native/calibrated anchor checks guard the new metric implementation. No other findings remain. Source/preflight review requires no additional GPU inference.

The root's `ROUND03_LAUNCH_AUTHORIZATION.md` supplies conditional execution authority. Monitor must first download and verify the preparation archive, then inspect staged files, commit and push, and issue the bound receipt. Immediate GPU admission follows. No further approval is needed after those gates pass. Subsequent results still require independent review and a root decision before further work.
