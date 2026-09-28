# Validation-only MTO eta ensemble diagnostic

## Protocol and alignment

Candidates were frozen in `PROTOCOL.json` before scores were calculated: the saved MTO eta=0, eta=0.1, and eta=1 validation predictions. Exact molecule ID and source row-index vectors, raw `f_true`, and masks match across all three sources; the ten prediction columns remain in their stored state-slot order. The split has 6,686 molecules × 10 state slots; five deterministic molecule-level folds were assigned by SHA256(ID) modulo five.

No test files were opened. c64G1 was excluded because no saved validation array was available; no checkpoint inference was run. Nonsupervised-energy models were excluded.

## Results

| Candidate | validation R² | SSE | RMSE |
|---|---:|---:|---:|
| mto_eta0 | 0.405294 | 94.205121 | 0.037537 |
| mto_eta01 | 0.377127 | 98.667021 | 0.038415 |
| mto_eta1 | 0.387367 | 97.044934 | 0.038098 |
| mto_eta0+mto_eta01_equal | 0.430543 | 90.205598 | 0.036731 |
| mto_eta0+mto_eta1_equal | 0.441906 | 88.405590 | 0.036363 |
| mto_eta01+mto_eta1_equal | 0.431197 | 90.101998 | 0.036710 |
| all_three_equal | 0.449421 | 87.215074 | 0.036117 |

Five-fold molecule-level simplex fit, scored only on held-out validation molecules:

- OOF R²: 0.443012; SSE: 88.230309; RMSE: 0.036327.
- Mean fold weights: {"mto_eta0": 0.3972416547456089, "mto_eta01": 0.25196695829018656, "mto_eta1": 0.35079138696420453}.
- Bright threshold (validation raw truth q90): 0.0546; OOF bright SSE: 57.656672 across 6690 transitions.

Residual Pearson correlations:

| |mto_eta0|mto_eta01|mto_eta1|
|---|---|---|---|
|mto_eta0|1.000000|0.871037|0.849092|
|mto_eta01|0.871037|1.000000|0.841520|
|mto_eta1|0.849092|0.841520|1.000000|

Per-state-slot SSE and bright OOF SSE are in `ensemble_metrics.json`.

## Interpretation

This diagnostic tests whether the saved eta MTO predictions contain enough complementary residual variation to justify a later frozen ensemble evaluation. The reported CV is conditional on fixed predictions: their checkpoint selection already used validation, so the CV scores are exploratory. A later untouched split is needed before claiming improved generalization. The diagnostic does not select or alter a production model.
