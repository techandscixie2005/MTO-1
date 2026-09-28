# Independent chan64 campaign closeout and fixed-ensemble review

Reviewed at: 2026-09-29T04:37:48.046827+08:00

## Decision

Close the present chan64 training campaign without extension or G2 resumption. Retain the fixed five-model native-f mean as a **provisional validation candidate**: the saved-array evidence shows useful complementary errors despite weak individual chan64 scores. This changes candidate priority; it does not establish a width/readout causal benefit, justify test access, or alter the four running studies. Any later independent-seed confirmation needs a frozen protocol; another query of the historically reused test would not be fresh confirmation.

## Verified selection and completion

G1/G3/G4 naturally early-stopped at epochs 227/326/223; their legacy joint-objective selected epochs are 75/174/71. G2 completed 214 epochs then was intentionally ADMIN_STOPPED within epoch 215. The supervisor marks the campaign blocked/partial. All four source fingerprints/configurations, recorded checkpoint/terminal hashes, data/scaler hashes, history continuity and minimum-joint selection were independently checked.

G3 selected joint loss 0.5248532128786919 is not oscillator-strength R². G1/G3 immutable joint-selected checkpoints were replayed exactly once on guarded healthy idle GPU4 with unchanged ECC. Primary f uses each model's predicted E times FP64 trace(A), multiplied by (2/3)/27.211386245988. Historical FP32-trace and primary scores both reproduce old aggregate R² within 5e-8, under the sealed 1e-6 gate. Exact validation IDs/indices, raw printed FP64 truth, masks and all 10 states match the five component files; native E/trace arithmetic reconstructs exactly. G2/G4 historical f values use oracle E and are excluded. No raw-f-optimal checkpoint selection is established by these legacy records.

## Full validation results

| Predictor | Raw-f SSE | Raw-f R² | ΔR² vs fixed equal3 |
|---|---:|---:|---:|
| eta0 | 94.205121 | 0.405294118 | −0.044127345 |
| G1 | 94.211383 | 0.405254589 | −0.044166874 |
| G3 | 100.096791 | 0.368100694 | −0.081320769 |
| Fixed equal3 | 87.215074 | 0.449421463 | 0 |
| Fixed G1/G3 equal2 | 84.893431 | 0.464077728 | +0.014656265 |
| Fixed eta0/eta01/eta1/G1/G3 equal5 | 81.151578 | 0.487699605 | +0.038278142 |

Five-way SSE falls 6.063496 (6.95%). Paired 2,000-draw molecule ΔR² CI is [0.026124, 0.055522]; connectivity-group CI [0.025972, 0.054697], with positive fraction 1.0. Two-way CIs cross zero: molecule [−0.011909, 0.050374], group [−0.012312, 0.050477], positive fractions 0.822/0.817. Independent array arithmetic and bootstrap implementation reproduce the reported results to numerical precision. These intervals describe sampling variation conditional on validation selection; they omit adaptive research/model-selection uncertainty.

## Distribution and tradeoffs

Five-way improves SSE for 4,091/6,686 molecules and nine states; S8 worsens by 0.067892 SSE, while S7 improves by 1.597448. Below-q90 improves by 4.864880 SSE and q90-and-above by 1.198616, but q99-and-above worsens by 0.623884 (30.387212 to 31.011095). Threshold ties are retained: q90=0.0546 has 6,690 transitions, q99=0.2377 has 671.

The largest molecule and connectivity-group contribution is ID 14562: SSE 3.596541 to 2.622175, gain 0.974366, or 16.07% of five-way net gain. The ten largest positive contributions sum to 2.129085 (35.11% of net). Gains are distributed beyond this case: all six frozen TRAIN-derived q2 bins improve in absolute SSE. The six lowest-q2 molecules improve 5.256918 to 4.141298, gain 1.115620; the two populous highest-q2 bins together contribute 4.626118 gain. The rare six-case stratum remains too small for a geometry-causality claim.

Two-way is more concentrated: ID 14562 supplies 87.89% of its net gain; S7 improves 2.137091 SSE but S8 worsens 1.634121, and q99 worsens 4.014522. Concentration and bright-tail deterioration are robustness diagnostics, not reasons to exclude valid cases or reject pooled gains. Every headline retains the full benchmark.

The five-way requires five model forwards versus three; two-way requires two. G1/G3 have 3.45M/3.39M parameters, so forward counts do not imply calibrated latency ratios. No weight/subset search was performed. The fixed two additions were sealed after individual results but before replay/combined scores; review hashes bind the protocol, and local timestamps support that chronology without claiming external notarization.

## Evidence and limits

Independent artifacts: `CAMPAIGN_INDEPENDENT_VERIFICATION.json`, `ENSEMBLE_INDEPENDENT_VERIFICATION.json`, `ENSEMBLE_CONCENTRATION_REVIEW.json`, and `REPLAY_REVIEW.json`; executor records: `CAMPAIGN_AUDIT.json`, `REPLAY_RESULT.json`, `ENSEMBLE_RESULTS.json`. Raw prediction arrays/checkpoints remain server-only. Geometry uses the pinned Round09 TRAIN cuts and reviewed helper; no bins were fitted to these outcomes. No new model inference by this reviewer, training, test evaluation, case filtering, or extra ensemble candidate was used. G2 incompleteness and joint-selection mismatch limit a broad architectural conclusion, while the fixed-five improvement is a concrete exploratory candidate worth preserving.
