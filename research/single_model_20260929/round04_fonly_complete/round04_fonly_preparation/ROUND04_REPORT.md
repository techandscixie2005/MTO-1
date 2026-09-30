# Round04 completed: fixed scalar flexibility did not improve accuracy

Status: both authorized CPU fits/export and the single prescribed comparison completed normally. No test scoring, new benchmark scoring, model averaging, checkpoint search or hyperparameter search occurred. No scientific source changed.

## Result and decision

Retain the existing single-checkpoint historical affine incumbent. Neither new four-coefficient map meets the frozen confirmation rule: a gain of at least 0.003 pooled raw-f R² over both its corresponding affine control and the incumbent. The held-out-source map exceeds the in-sample map by 0.01344438, but this source contrast does not establish a benefit from added scalar flexibility. No extension, refit or seed allocation is recommended for these maps.

All rows below use the same 6,686 historically reused validation molecules and all 66,860 raw printed-f labels. This is a legacy diagnostic, not fresh holdout confirmation and not evidence for attaining R² 0.60 on a new benchmark.

| Predictor | Pooled raw-f R² | RMSE | MAE | SSE |
|---|---:|---:|---:|---:|
| Native eta0 | 0.405294124 | 0.037536525 | 0.017785608 | 94.205120 |
| Historical validation affine (incumbent) | 0.418119244 | 0.037129573 | 0.018219066 | 92.173541 |
| In-sample affine | 0.402024351 | 0.037639574 | 0.017819879 | 94.723072 |
| Held-out-source affine | 0.416374515 | 0.037185196 | 0.017805840 | 92.449917 |
| In-sample fixed hinge | 0.398331068 | 0.037755632 | 0.017839519 | 95.308112 |
| Held-out-source fixed hinge | 0.411775449 | 0.037331421 | 0.017862327 | 93.178438 |

The in-sample hinge loses 0.00369328 R² against its affine control; the held-out-source hinge loses 0.00459907 against its affine control and 0.00634379 against the incumbent. Their descriptive paired-molecule bootstrap intervals versus their affine controls include zero; the direction of the point estimates supplies no promotion. These intervals reuse validation and do not correct for model selection.

## All-state comparison

| State | Native | Incumbent | In affine | Held affine | In hinge | Held hinge |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.827284 | 0.819549 | 0.825396 | 0.827166 | 0.825325 | 0.824347 |
| S2 | 0.735728 | 0.735512 | 0.733034 | 0.739980 | 0.735698 | 0.737484 |
| S3 | 0.590072 | 0.598698 | 0.586662 | 0.599505 | 0.588617 | 0.599286 |
| S4 | 0.390116 | 0.412284 | 0.385316 | 0.408294 | 0.385933 | 0.404217 |
| S5 | 0.420834 | 0.411932 | 0.420862 | 0.416277 | 0.424243 | 0.424380 |
| S6 | 0.243128 | 0.271754 | 0.238173 | 0.264126 | 0.232113 | 0.258953 |
| S7 | 0.049940 | 0.121369 | 0.039267 | 0.100106 | 0.013278 | 0.068374 |
| S8 | 0.306930 | 0.304911 | 0.306185 | 0.307072 | 0.304450 | 0.305396 |
| S9 | 0.301463 | 0.320691 | 0.297735 | 0.316069 | 0.288927 | 0.311398 |
| S10 | 0.150100 | 0.160428 | 0.148375 | 0.157048 | 0.142089 | 0.151007 |

All states contain 6,686 labels. The held-out-source hinge improves only S5 against its own affine control; its largest SSE regression is S7 (about 0.464). Per-state SSE, RMSE and MAE for every predictor are retained in VALIDATION_COMPLETE.json.

## Bright-tail and false-bright errors

Thresholds were fixed from the historical TRAIN: q90=0.0549 and q99=0.2412. True q90 contains 6,650 labels and true q99 contains 639. False-bright means true f<q99 and predicted f>=q99; no bin is excluded.

| Predictor | q90 RMSE | q99 RMSE | q99 MAE | False-bright count | False-bright SSE |
|---|---:|---:|---:|---:|---:|
| Native eta0 | 0.095359 | 0.219913 | 0.163887 | 112 | 12.603675 |
| Historical validation affine (incumbent) | 0.098349 | 0.231384 | 0.178882 | 69 | 8.418992 |
| In-sample affine | 0.095038 | 0.218778 | 0.163034 | 123 | 13.320506 |
| Held-out-source affine | 0.097486 | 0.227282 | 0.172683 | 86 | 9.716859 |
| In-sample fixed hinge | 0.096065 | 0.221038 | 0.166151 | 111 | 13.760531 |
| Held-out-source fixed hinge | 0.097783 | 0.229272 | 0.176597 | 70 | 10.108559 |

The held-out hinge has fewer false-bright predictions than its affine control (70 versus 86), but greater false-bright SSE (10.10856 versus 9.71686) and worse true-tail errors. Counting bright predictions alone would conceal this tradeoff. Every predictor's four brightness-bin counts sum to 66,860 and their SSE sums to pooled SSE.

## Fitting and application diagnostics

Each map used exactly one unconstrained FP64 least-squares solve on the same 24,071 calibration molecules and 240,710 valid labels. The fixed basis is [1, f/s, max(0,(f−0.0549)/s), max(0,(f−0.2412)/s)], s=0.050109692115345626. The loss is unclipped raw-f squared error; deployment applies max(0, q). No energy, state index, hidden features or QC labels enter the map.

| Source | Rank | Condition number | Affine fitting SSE (unclipped) | Hinge fitting SSE | Above upper knot |
|---|---:|---:|---:|---:|---:|
| in_sample | 4 | 10.319998 | 213.875927 | 213.131609 | 1600 |
| heldout_source | 4 | 9.651780 | 357.033701 | 356.131163 | 1252 |

Fitting SSE improves as expected for the nested basis; transfer accuracy does not. This does not establish a causal reason for the failure. Both fitted maps have nonnegative segment slopes. Neither clamps any calibration or validation output. All validation inputs lie inside each corresponding source fitting range; 348 validation inputs are at or above q99. No rank/condition fallback, knot adjustment or exclusion was used.

Both deployment files contain the same full eta0 backbone and one scalar map. The 80%-TRAIN source is used only to obtain calibration predictions and is absent at inference. E and A are unchanged by the scalar map, so calibrated f generally no longer satisfies the model's original E/trace(A) identity. Physical interpretation of the scalar map is unsupported.

## Execution, parity and provenance

The CPU environment was bound before Python imports: CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2. A live owned wrapper was registered before each child, with PID/start tick/boot ID/UID/cwd/argv identity in its receipt. Fit/export took 14.40 seconds and evaluation 4.79 seconds; both exited 0 and verified authorization unchanged. There was one fit/export attempt and one validation attempt.

Both first-64 calibration geometry replays passed: native/cache maximum absolute difference 8.776325466364199e-7, consistent with FP32 batching differences; scalar-map arithmetic difference 2.220446049250313e-16. Backbone tensors were unchanged. Coefficients were frozen before validation. No new backbone forward was needed for validation; the declared scalar comparison used the frozen Round03 native prediction cache.

Frozen source manifest: `79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f`. Execution authorization: `530db555da8badefb773b945e16f5e5ccce9c54fc796f78ec0d6a420b9dc6b97`. Exact aggregate and operational input hashes are in ROUND04_TERMINAL_MANIFEST.json. The independent terminal review is separate.

## Private checkpoint locations

- in_sample: `/home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation/production/in_sample.pt`; SHA256 `f43897f1bc23b2981d617f3f0bd11fbc84499f6c2d162eb13227934ef989981f`.
- heldout_source: `/home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation/production/heldout_source.pt`; SHA256 `5330d4028f460043883c08f66c4653cf75f0cb5a82a8716c7554b8f2b15619af`.
- Retained incumbent: `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`; SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.

All learned tensors, checkpoints, raw arrays and prediction caches stay on the server. Lightweight records contain settings, aggregate diagnostics, shapes and hashes, with no learned coefficient values.

## Reproduction and safe continuation

The exact reviewed commands below document the completed run. Current completion markers deliberately block rerunning these stages; do not remove them. A distinct experiment requires separate authorization.

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python ops/run_cpu_stage.py fit_export
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python ops/run_cpu_stage.py evaluate
```

The wrapper resolves the pinned source manifest, independent preparation review, preparation publication receipt and PRODUCTION_EXECUTION_AUTHORIZATION.json, and verifies their exact binding before the unchanged scientific entry point. No further scientific execution is pending for Round04. The remaining task is independent terminal review, archive-first publication and the requested repository README update.
