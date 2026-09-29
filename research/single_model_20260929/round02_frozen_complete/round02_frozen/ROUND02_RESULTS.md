# Frozen-F objective pilot

Each row is one model and one checkpoint. Original parameters and all buffers stayed frozen. No test inference or prediction averaging. Validation is reused; independent-seed confirmation remains necessary.

| Arm | Selected epoch | Native R2 | Delta vs epoch0 | Fixed calibration R2 | Epoch20 native R2 |
|---|---:|---:|---:|---:|---:|
| trace | 2 | 0.405407830 | +0.000113700 | 0.417538722 | 0.403125542 |
| raw_f | 1 | 0.405387525 | +0.000093404 | 0.417865155 | 0.402157980 |

Selected raw-f minus trace R2: -0.000020306.
Priority over the zero-update anchor (>=.003): none.
Raw-f objective also passes >=.003 against the active trace control: False.

Every state, true-bright tail, selected true/predicted brightness-bin partition, train loss/movement/clip trajectory, exact checkpoint path/hash and descriptive bootstrap interval appears in ROUND02_RESULTS.json.
F changes both E and A through the frozen decoder. Fixed calibration is secondary and was never refit. Base representations remain in-sample for TRAIN; freezing does not remove inherited overfit.
The raw-f coefficient1 gives a different gradient scale from the trace objective. No scale or schedule was adapted to validation. The shared base_objective field means LE+Ls; objective-specific validation values are computed explicitly in aligned records.
