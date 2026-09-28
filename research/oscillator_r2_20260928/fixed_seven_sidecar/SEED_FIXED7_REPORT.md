# Fixed-seven saved-array validation sidecar

Mode: seed family completed. Every pooled score retains all 6,686 molecules and 66,860 raw-f labels.
This is a saved-array calculation; no model inference, training, fitting or test access.

## Pooled raw-f metrics

| Predictor | SSE | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| F5 | 81.151577883 | 0.487699605 | 0.0348390 | 0.0163345 |
| seed3_primary | 86.655328163 | 0.452955075 | 0.0360010 | 0.0170211 |
| F7 | 81.311355006 | 0.486690951 | 0.0348732 | 0.0163709 |

## F7_minus_F5

ΔR² -0.001008654; ΔSSE +0.159777123.
Molecule CI [-0.020300527468791556, -0.00022038459592339663, 0.010452893886135642]; connectivity-group CI [-0.02051874411058313, -0.00018673044897409957, 0.010437931285063064].
Per-state, fixed-tail and concentration details are in JSON.

## F7_minus_seed3_primary

ΔR² +0.033735876; ΔSSE -5.343973158.
Molecule CI [0.014627632368699118, 0.03228350214347733, 0.06910504555160718]; connectivity-group CI [0.014659339630814634, 0.03280286305386112, 0.06788354871100427].
Per-state, fixed-tail and concentration details are in JSON.

Forward counts: F7=7, F5=5, seed3_primary=3. These are not calibrated latency ratios.
The seed3 primary and mixed-selection secondary in the original seed study remain unchanged.
F5 is a provisional validation candidate; the original fixed-three historical-test score belongs only to that predictor.
All intervals are descriptive conditional on reused validation selection.
