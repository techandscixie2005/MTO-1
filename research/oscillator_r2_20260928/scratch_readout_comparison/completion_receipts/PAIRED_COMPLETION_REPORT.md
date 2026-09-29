# Completed paired scratch readout screen

Both prespecified arms finished 100 epochs and 188,100 optimizer steps with the same seed11 fresh backbone and minibatch order. Model checkpoints were selected by validation raw native-f SSE among epochs 0–100. The comparison below uses saved validation predictions, raw FP64 f labels and all 6,686 molecules (66,860 transitions). No audit-time model inference or test evaluation was run.

| Predictor | Selected epoch | Validation SSE | Validation R² | Selected full-train R² | Final100 validation R² |
|---|---:|---:|---:|---:|---:|
| native direct f | 21 | 103.949027936 | 0.343781976 | 0.584645710 | 0.227522435 |
| mto direct f | 16 | 99.242624587 | 0.373492948 | 0.517470659 | 0.276909374 |

The primary native-minus-MTO validation delta is -0.029710972 R², or +4.706403348 SSE. Paired 2,000-resample 95% intervals are [-0.05285466961222263, -0.007184501261827185] by molecule and [-0.0549249420005203, -0.006530600809318478] by connectivity group. These intervals are descriptive because validation selected checkpoints and the same split has informed prior research.

The prespecified post-launch, pre-outcome 50/50 mean of the two selected native-f arrays reaches R² 0.403133570, SSE 94.547365302. Context: eta0 R² 0.405294118; original fixed equal-three R² 0.449421463; provisional five-model validation candidate R² 0.487699605. The scratch arms and their mean do not supersede that candidate. The five-model test score remains unevaluated.

At epoch100, native train R² 0.914208 versus validation R² 0.227522; MTO train R² 0.905996 versus validation R² 0.276909. Both trajectories show overfitting in this fixed setup. The selected MTO direct readout is better than the selected native DetaNet direct readout here; this does not isolate backbone architecture or establish a universal ranking.

The reviewed geometry report keeps all rows, uses frozen train-derived q2 bins, and treats q3 descriptively. It records physical-state SSE, signed errors, q90/q99 tails and case14562/six-case error concentration. No new bins, sample filters, weights or model calls were introduced.

## Provenance

- Scratch primary summary: `scratch_readout_comparison/RESULTS.json` SHA-256 `827763806c6b4f75e2c915eaf090ec4b480b79a0800db8296475c2b845dda665`; readable report `RESULTS.md` SHA-256 `52f53c79cf1f2ad57ff58a3edeb1e70488219dc0d471bd8aea29ed102eb3342e`.
- Reviewed geometry output: `postrun_geometry/SCRATCH_RESULTS.json` SHA-256 `748efebdb2046c9e339d39f520d11262108d0c12f1c711814b428f8f1c14b710`; readable report `SCRATCH_REPORT.md` SHA-256 `ded9bbfc158b97395ae4d91497c4887e622e6045f858cd8dc8f8649357c32240`.
- Native terminal `runs/native/FIT_COMPLETE.json` SHA-256 `31ade59511a4fde1b323201de387ae873c31f83d5dfa49f595faae7cc5b31abb`; MTO terminal `runs/mto/FIT_COMPLETE.json` SHA-256 `7d8838482001cbcafe9e29a059daa6ce20127debf0e518e3a437bd2594a488e3`.
- Scratch protocol SHA-256 `9be3d271f86afac91a2b20957b4edee86bad6a14b0bd9058c564254ad8b151b2`, implementation review SHA-256 `6226b1c68226ce8a5ed6ddf74c630bce32dd88e0fadc46b0de12e8e4d005c3e2`, and postrun implementation review SHA-256 `aeb1363fdb211b7306751d6c48de09b7c3bd62f48915d0d117c204e9a322ed6b`.
- Runtime selected NPZ/checkpoint files and their hashes remain on the server in immutable terminal receipts. They are excluded from the D/GitHub lightweight archive.
