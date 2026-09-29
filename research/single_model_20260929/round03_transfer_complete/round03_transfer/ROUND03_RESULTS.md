# Round03 affine transfer result

All fixed 33 source epochs completed. Both affine pairs were frozen before outer validation; no test inference or prediction averaging occurred.

| Predictor | Pooled raw-f R² | RMSE | MAE | q90 RMSE | q99 RMSE |
|---|---:|---:|---:|---:|---:|
| native | 0.405294124 | 0.037536525 | 0.017785608 | 0.095359275 | 0.219912917 |
| historical_validation_fit | 0.418119244 | 0.037129573 | 0.018219066 | 0.098349265 | 0.231384276 |
| in_sample | 0.402024351 | 0.037639574 | 0.017819879 | 0.095038371 | 0.218778366 |
| heldout_source | 0.416374515 | 0.037185196 | 0.017805840 | 0.097485595 | 0.227281574 |

| Map source | Alpha | Beta | Source prediction R² on the same 24,071 molecules |
|---|---:|---:|---:|
| in_sample | 1.018095001122 | -0.000290934645 | 0.661066543 |
| heldout_source | 0.902585325224 | 0.001828558211 | 0.429408124 |

## Per-state validation R²

| State | Native | Historical affine | In-sample affine | Heldout-source affine |
|---|---:|---:|---:|---:|
| 1 | 0.827283935 | 0.819548742 | 0.825396043 | 0.827165913 |
| 2 | 0.735727566 | 0.735512419 | 0.733033512 | 0.739980205 |
| 3 | 0.590071513 | 0.598698120 | 0.586661874 | 0.599504591 |
| 4 | 0.390116141 | 0.412283867 | 0.385316450 | 0.408294015 |
| 5 | 0.420834386 | 0.411931721 | 0.420861961 | 0.416276964 |
| 6 | 0.243128001 | 0.271753814 | 0.238173021 | 0.264126319 |
| 7 | 0.049940142 | 0.121368583 | 0.039267387 | 0.100106153 |
| 8 | 0.306929802 | 0.304910723 | 0.306184814 | 0.307071957 |
| 9 | 0.301463310 | 0.320690798 | 0.297734771 | 0.316068832 |
| 10 | 0.150100419 | 0.160428343 | 0.148374967 | 0.157047641 |

## Interpretation boundary

Heldout minus in-sample R²: 0.014350164; heldout minus native: 0.011080390. The prespecified nonlinear-preparation allocation gate is passed.

The source uses fewer fitting molecules and different optimization errors. This comparison tests this one transfer procedure, not a causal explanation of generalization. The historical affine was fitted with exposed validation information. Outer validation has been reused; this is not fresh holdout confirmation. Both deployed maps use the same full baseline and may be inconsistent with its unchanged auxiliary E/A outputs.

Source online fitting losses describe a changing model. Every source checkpoint is fixed at epoch33; no calibration or outer label selected it. All 33 prescribed orders and 49,665 optimizer updates were verified. Full optimizer/RNG states and every model/prediction array remain on the server.

Resource limitation: a short-lived MTO process was observed on GPU0 during the affine-fit interval; exact argv and full inference placement were not captured before it exited. The fit was retained without repeat. Validation was prebound before interpreter startup and actually observed only on GPU1. See RESOURCE_PLACEMENT_NOTE.md and the owned validation launch receipt.

ROUND03_RESULTS.json contains complete pooled/state/tail errors, all-label brightness partitions, coefficients, source quality, runtime and checkpoint metadata. Root decides the next action after independent review.
