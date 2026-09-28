# Geometry error audit protocol

Status: fixed validation-only diagnostic; no fitting, inference, sample removal, or test access.

## Question and fixed case

Determine whether validation molecule ID 14562 (composition C8H2; all ten raw oscillator strengths reported as zero) is part of a broader geometry-associated error population. The case may be quantified for concentration analysis, but all aggregate benchmark metrics retain every validation molecule.

## Pinned inputs

- Geometry and split IDs: `experiments/qm9s_eta_Ef_20260926/data/dataset.npz`; use only its `train` and `val` rows, `z`, and `pos` arrays.
- Raw labels and masks: `experiments/qm9s_eta_Ef_20260926/data/raw_labels.npz`; index and analyze only the train/validation indices.
- Validation predictions: saved `val_predictions.npz` for `mto_eta0`, `mto_eta01`, and `mto_eta1`. Equal3 means the arithmetic mean of these three already-saved native-f arrays.
- Existing connectivity grouping: second field of `experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json`, aligned by dataset row index and checked against molecule ID.

All IDs, source indices, raw FP64 f labels and masks must match exactly before analysis. No test indices, labels, geometries, predictions, or metrics are read.

## Geometry and strata

For each molecule, select non-padding atoms (`z != 0`), convert coordinates to FP64, subtract the unweighted arithmetic coordinate mean, and form the centered unweighted covariance `X.T @ X / n_atoms`. Sort its symmetric eigenvalues as `lambda1 >= lambda2 >= lambda3 >= 0`. Define `q2=lambda2/lambda1` and `q3=lambda3/lambda1`. If `lambda1 <= 64 * eps64 * max(trace(covariance), 1)`, mark the ratios undefined and report the molecule in a separate degenerate-geometry count. Fixed near-linear classification is `q2 <= 1e-5`.

Use the finite training-set `q2` distribution only to calculate quantiles 0.001, 0.01, 0.10, and 0.50. Form sorted unique cut points from those four values plus the fixed `1e-5` threshold. Report consecutive intervals from zero through each cut point, plus the upper tail; state right-closed interval convention and duplicate cut-point removal. The same boundaries classify validation. Do not tune or search thresholds.

## Outcomes

For eta0 and equal3, report by validation shape stratum and state: molecule/label counts; exact-zero raw-label prevalence; pooled SSE, fraction of all validation SSE, mean SSE per molecule; mean signed prediction error and fraction overpredicted. Also report molecule 14562's geometry and per-state errors/SSE contribution; formula, near-linear, and exact connectivity-group matches in training, with counts and IDs. Preserve the full-sample benchmark; any case-exclusion arithmetic is labeled concentration-only and must not replace the benchmark.

## Execution constraints

CPU only, at most two CPU threads. No model loading, new inference, fitting, test access, or editing of sealed experiment sources. Write protocol, code, JSON, and report under `research/oscillator_r2_20260928/geometry_error_audit/` using explicit UTF-8. Record source hashes and execution timestamp. Read back the completed files and check substantive content, hashes, and absence of placeholders before handoff.