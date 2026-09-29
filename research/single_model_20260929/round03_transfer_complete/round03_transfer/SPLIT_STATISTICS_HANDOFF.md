# Round03 internal split and source normalization

Preparation completed on CPU. No model inference, fitting or GPU was used. The original outer split is unchanged.

## Exact grouping contract

The generator reads original TRAIN indices and the existing identity audit. It shuffles sorted resolved group keys with NumPy `default_rng(20260930)`, takes whole groups until at least 24,071 calibration rows are assigned, and preserves original TRAIN order within both subsets. All 266 unresolved groups (278 rows) stay in source fitting. No group is divided and no original TRAIN molecule is excluded.

- Source fitting: 96,284 molecules, index-byte SHA `b7b00dfe514ae57f3dd609a60dc259625007a5f43288ff75e2c83a3a7e71da37`.
- Calibration: 24,071 molecules in 24,032 groups, index-byte SHA `9547ca89feff3301f5e4c6fcd736be7a9186878a9fcb1fec8695c5c12d60707b`.
- Identity-audit SHA `9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4`.

`split_manifest.json` records source hashes, group counts and both file hashes and index-byte hashes. The `.npy` arrays and identity table remain server-only. Re-running the generator accepts only identical existing arrays.

## Fit-only statistics

Only source-fit E, A and their masks are decoded from the FP64 raw-label file; all 962,840 state labels are valid for each target. The generator reads fit atomic numbers for atom counts and ID metadata for alignment. It does not decode f or calibration/outer-validation/test targets. ZIP streams may pass nonselected compressed bytes, which are not interpreted as target rows or retained in arrays.

- `E_state_mean`: ten separate means over fitting molecules.
- `sE2 = mean((E_fit - E_state_mean)^2) = 0.5384852554041385`.
- `sA2 = mean(sum_ij(A_fit_ij**2)) = 0.13127702708928576`.
- `n_ref = median(fit atom count) = 18`.

These reproduce the original reduction definitions using the new fitting subset. An independent second-moment identity and a blockwise tensor sum check the reductions. The training loader uses original FP32 dataset E/A, as the original recipe did; normalization is computed from raw FP64 labels before that cast.

`STATISTICS_AUDIT.json` contains exact access scope, formula checks, dependency hashes and output hashes. `fit_normalization.json` retains the original constructor keys plus explicit source-scope metadata.

## Initialization and selection constraints

Build a fresh `MTOEA(source_config, fit_normalization)` using torch seed11. `model_factory.build` in the original experiment constructs this architecture from random parameters. The original `initial_model.pt` embeds the full-TRAIN energy means and must not be reused for this source; the full trained eta0 checkpoint is likewise forbidden as source initialization. Source training may resume only its own pinned state.

Fit exactly 33 epochs using original LE+Ls and the fixed original optimizer settings. No calibration, outer-validation or test label can affect initialization, normalization, gradients, scheduling or source selection. The 20% calibration set becomes available only after the fixed epoch33 source is sealed. A smaller source has different training size and quality; this is not full cross-fitting or fresh outer holdout confirmation.

## Reproduction

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  /home/inspur/MTO-1/research/single_model_20260929/round03_transfer/prepare_split_statistics.py
```

Pin `prepare_split_statistics.py`, the grouping helper `../clean_source_feasibility/audit_clean_source.py`, and the selective NPZ reader `../reports/velocity_audit/audit_velocity_labels.py` in the source closure. This handoff is preparation evidence, not a fit-approval receipt.
