# Fixed descriptor probe: independent closeout

Recorded at 2026-09-29T07:51:07.296839+08:00. **Completed-artifact review PASS; close this fixed probe without retuning, extension, additional weights or test access.** F5 remains the provisional validation candidate.

## Outcome and decision

| Predictor | Full validation SSE | Pooled raw-f R² | MAE |
|---|---:|---:|---:|
| Fixed descriptor forest D | 124.423762 | 0.214527379 | 0.021453695 |
| Prescribed half-D/half-F5 | 93.215196 | 0.411543397 | 0.018462054 |
| Frozen F5 | 81.151578 | 0.487699605 | 0.016334466 |

D full-TRAIN R² is 0.486264044 (SSE 1552.557053), considerably below the later neural TRAIN fits. There is both a training/validation gap and modest training fit. Fixed depth, leaf size, random feature subsets, representation loss and their interaction can all limit this one forest. This result does not reject descriptor methods or nonlinear invariants generally, identify a backbone ceiling, or justify a retrospective weight search.

Independently reproduced paired 2,000-draw bootstrap results (seed 20260929, ten states together, pooled SST recomputed; groups sorted by connectivity key): D minus F5 ΔR² −0.273172226, group 95% interval [−0.302444048, −0.243225761]; half blend minus F5 −0.076156208, group interval [−0.087926190, −0.062927143]. Both positive fractions are zero. Corresponding molecule intervals are [−0.303008657, −0.243435087] and [−0.088689362, −0.063099538]. These are descriptive under reused validation and adaptive study choice, not fresh confirmation.

## Where the fixed blend loses

All ten state SSEs worsen against F5; the largest increases are S2 +2.227205, S5 +1.983391 and S3 +1.756790. Below-q90 SSE improves from 24.924465 to 22.343225 (gain 2.581239), but q90-and-above worsens from 56.227113 to 70.871971 (loss 14.644858), yielding net SSE loss 12.063618. Within the bright tail, q99 worsens by 11.126217. D alone likewise loses chiefly on bright targets: +42.083174 of its +43.272184 total SSE deterioration occurs at or above q90. An almost zero overall signed error for D (+0.000031532) does not prevent large squared error; the half blend mean signed error is −0.001013305 versus F5 −0.002058142.

For the blend, 2,439 molecules improve and 4,247 worsen; gross gains 3.835472 are outweighed by losses 15.899090. Largest positive molecule ID129198 gains 0.286816 SSE; largest negative molecule/group ID3024 loses 0.594715. ID14562 improves modestly (2.622175→2.367665 SSE), yet the six lowest-q2 molecules collectively worsen (4.141298→4.654100). Every other fixed q2 bin worsens, including the next five cases (1.519123→2.286435). Thus this is not a general solution to the small near-linear stratum. One and five molecules cannot establish geometry causality; all rows remain in the headline benchmark. No gain share is reported for a negative net gain.

## Integrity, phase order and measured cost

I checked every reviewed source hash, the exact authorization/review/seal chain, full TRAIN/validation feature and prediction hashes, and the fitted forest metadata without invoking fit or predict. Feature tables are finite FP32 with shapes 120355×803 and 6686×803; the original 16 preflight TRAIN fingerprints exactly match their stored rows. All 256 trees have the prescribed parameters, 803 inputs, ten outputs, 120355 samples at their roots, depth at most 24 and at least four samples in every leaf. The full estimator contains 4,673,848 nodes and is 673,133,769 serialized bytes.

Saved TRAIN truth exactly matches raw FP64 labels at all fixed TRAIN indices, with valid masks and the frozen TRAIN-ID hash. I recomputed its mean, common target scale 0.050109692115345626 and all fit metrics. Validation has all 6686 exact IDs/indices and 66860 raw FP64 labels/masks. The five component hashes reproduce F5; the stored blend is exactly (D+F5)/2. Both prescribed bootstrap contrasts, state/tail/q2/concentration and signed-error arithmetic reproduce independently.

FIT_FREEZE predates VALIDATION_STARTED, which binds its exact hash. PROBE_COMPLETE binds both successful phase receipts and results; neither phase has a resource failure, and no FAILED/INCOMPLETE_RESOURCE marker exists. Total elapsed time was 138.54 seconds. TRAIN feature construction took 89.77 s, fit 26.45 s and TRAIN prediction 1.64 s; validation features took 5.07 s and prediction 0.124 s. Peak sampled phase RSS was about 1.46/1.53 GiB, well below 16 GiB. Admission had 506.34 GB (471.6 GiB) available, eight selected idle logical CPUs, with hard 16-GiB address-space and four-hour total limits in the reviewed child/supervisor. The process had already completed before the requested live /proc check: enforcement is supported by the reviewed child gates and genuine successful phase receipts, not a retrospective claim to live-process inspection.

Standalone inference costs CPU feature construction plus one 256-tree estimator. The blend adds five neural forwards; this is not six equal-cost neural passes. Runtime arrays, feature tables and forest stay server-only. This audit performed no model inference, new features, fitting, active scratch inspection or test access.

## Bound evidence

INDEPENDENT_COMPLETION_AUDIT.json and independent_completion_audit.py contain the independent calculations and exact hashes. The companion SCIENTIFIC_PROVENANCE.json also pins executor completion evidence and the immutable terminal/fit/validation/resource receipts. A completed implementation with a negative predictive result is distinct from a resource failure; no rerun is needed.
