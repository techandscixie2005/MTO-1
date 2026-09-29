# Bounded velocity-label reconstruction audit

2026-09-30. **This is a training-label audit, not a prediction result or authorization for an auxiliary fit.**

## Scope and provenance

Uniform deterministic sample of4,096 IDs from the frozen120,355-molecule TRAIN split, RNG seed20260930. All40,960 sampled states were retained. `VELOCITY_TRAIN_AUDIT.json` records the sample-ID hash, sampled array-shard hashes, extractor/manifest hashes and exact source code hash. No molecule-level labels or prediction arrays are exported.

Only `ids` and `train` metadata were read from the curated dataset. For each extraction shard, labels were decoded only at the selected TRAIN row positions. Compressed ZIP streams necessarily pass nonselected bytes, but no validation/test label row was interpreted as numbers or analyzed. No model inference, GPU, raw Gaussian log retrieval, new QC calculation, or magnetic-label analysis occurred.

## Coverage in this sample

| Check | Count |
|---|---:|
| Present states, finite positive excitation energy, finite length f | 40,960 / 40,960 |
| Finite velocity vectors and printed velocity f | 40,960 / 40,960 |
| Valid source normal termination/convergence/single calculation/geometry | 4,096 / 4,096 molecules |
| Printed velocity f equal to zero | 680 |
| Velocity vector with all three printed components zero | 56 |
| Negative printed velocity f | 0 |

The extractor initializes optional fields with NaN and fills them when the labeled velocity table has the state index. No default-zero imputation was used. Physical/rounded zeros were retained. The original scalar/vector masks concern length-gauge targets, so this audit separately requires finite optional fields. Sample coverage does not establish whole-TRAIN coverage or resolve optional-table duplication/ambiguity beyond the existing parser.

## Scalar reconstruction

Tested `fV_reconstructed=(2/3)*sum(p_AU**2)/(E_eV/27.211386245988)` against the **printed velocity-gauge f**, using the preserved velocity-table components. The Gaussian table heading says atomic units. The same scalar convention is implemented by [PySCF's official oscillator-strength routine](https://pyscf.org/_modules/pyscf/tdscf/rhf.html). PySCF represents real-system velocity transition components as an imaginary part and contracts X−Y; this does not prove an identical Gaussian sign convention.

| Reconstruction discrepancy | Value |
|---|---:|
| MAE | 2.6981432e-5 |
| RMSE | 3.2361232e-5 |
| Maximum absolute residual | 1.8221888e-4 |
| Absolute residual≤1e-4 | 40,862 / 40,960 |
| Printed-value rounding intervals overlap | 40,960 / 40,960 |

All E, velocity components and printed fV lie on a1e-4 grid (maximum scaled floating-point grid error1.46e-11). The interval test permits±5e-5 in each printed component, E in eV, and fV. Lower/upper component-squared norms and inverse-energy bounds overlap every printed-f interval. Residuals slightly above1e-4 therefore do not contradict the stored precision. This supports **magnitude/unit consistency in the sample**, not vector phase alignment, full TDDFT amplitude reconstruction, or exact physical gauge equivalence.

## Length-versus-velocity discrepancy

Both strengths are existing QC outputs. Comparing them is a label-consistency diagnostic, not a model score.

| fV−fL diagnostic | Value |
|---|---:|
| Mean signed difference | −.0006506348 |
| MAE | .0016801611 |
| RMSE | .0039307701 |
| Maximum absolute difference | .0754 |
| 99th percentile absolute difference | .0178 |
| Agreement R² relative to fL | .9936272325 |

For the36,284 labels with fL≥.001, median absolute relative difference is7.3913%, mean13.0998%, p99=100%; this threshold only stabilizes the relative denominator. All40,960 present labels remain in absolute diagnostics. Differences are large compared with scalar reconstruction rounding, so do not replace length f by velocity f or assume equality.

## Decision

Velocity labels are a plausible existing auxiliary-information source, conditional on a complete TRAIN-only coverage/ambiguity audit and a separately reviewed objective. This bounded audit supplies a concrete scalar reconstruction check. It does not show that auxiliary training will improve raw length-f accuracy. Do not directly add length and velocity vectors, infer relative phases from their norms, impose a physical shared-response identity, or require these QC labels at inference. No auxiliary objective or fit is selected here.

Deferred hypothesis, ranked behind completed-factorial review and frozen-adapter diagnosis: a single geometry-only model could receive phase-insensitive fV or p p† auxiliary supervision, possibly through a shared length/velocity readout. A finite-basis relation between fL and fV would be approximate, not an exact equality constraint. The observed .00393 discrepancy RMSE is a label difference, **not a model noise floor**. Any such study requires the full coverage/mask/convention audit, fixed paired controls and unchanged raw-fL evaluation. It would not reconstruct X/Y, transition densities or NTOs.

Reproduce on CPU:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  /home/inspur/MTO-1/research/single_model_20260929/reports/velocity_audit/audit_velocity_labels.py
```

Only this report, its script and aggregate JSON belong in the next lightweight archive. Preserve the frozen original experiment and data splits.
