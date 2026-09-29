# Fixed scratch-seven saved-array secondary

Frozen equal seven individual native-f predictions; all 6,686 molecules and 66,860 states retained. No inference, fit or test.

| Predictor | SSE | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| F5 | 81.151577883 | 0.487699605 | 0.0348390 | 0.0163345 |
| F7_scratch | 82.277078236 | 0.480594453 | 0.0350797 | 0.0166435 |
| F7_seed_legacy | 81.311355006 | 0.486690951 | 0.0348732 | 0.0163709 |

## F7_scratch_minus_F5

ΔR² -0.007105152; ΔSSE +1.125500353.
Molecule CI [-0.01275922380817473, -0.0070474673997530475, -0.0024019364463624036]; connectivity-group CI [-0.012543477705632142, -0.006965557375125759, -0.002559569799253474].

## F7_scratch_minus_F7_seed_legacy

ΔR² -0.006096498; ΔSSE +0.965723230.
Molecule CI [-0.017134851209341818, -0.00707908398051286, 0.01048523307431584]; connectivity-group CI [-0.01729203016855179, -0.0070083420244428235, 0.010563390474671388].

Per-state, fixed tail, signed error and absolute concentration records are in JSON.
Native-minus-MTO remains the original scratch primary contrast. F5 and F7_seed are fixed supplemental references.
Validation has been reused; intervals are descriptive and no test is opened. Seven model forwards are not calibrated latency.
