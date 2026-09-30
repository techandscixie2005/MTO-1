# Round05 preparation results

**The bounded engineering checks passed. Production has not started; validation and TEST targets remain sealed during preparation.** These results establish implementation readiness, not an accuracy improvement.

## What is prepared

Four fresh models share the original MTO backbone, new TRAIN statistics, exact base initialization and data orders. The factors are the existing shared right-operand F and weak raw-M decorrelation. Their effects are tested jointly from initialization, unlike the earlier continuation/frozen-backbone experiments. Original LE+Ls and all common training settings remain fixed.

| Arm | Active F | Raw-M penalty |
|---|---:|---:|
| Control | No | 0 |
| Adapter | Yes | 0 |
| Decorrelation | No | .001 |
| Both | Yes | .001 |

The base has1,552,092 parameters and F adds4,016. Inactive F parameters are frozen; controls are not active parameter-count controls. Physical interpretations remain hypotheses. No QC labels or historical model predictions are required at inference.

## Evidence

| Check | Result |
|---|---|
| New split | Independent reconstruction passes;120355/6686/6686 rows, zero overlap under audited rules |
| Target boundary | Actual numeric decode ledger restricted to new TRAIN; no whole-array target load followed by slicing |
| TRAIN statistics | All1,203,550 E/A/f labels retained, including22,585 f zeros; new q90=.0549/q99=.2406 |
| Fresh initialization | Original/control/F inherited tensors and full wrapper schema match exact frozen hashes |
| Synthetic mathematics | CPU identity, rotations/reflections, permutations/translations and masked/zero-state penalty checks pass |
| One checkpoint | Load-time and forward-time external-data guards pass; all parameters/config/stats/buffers embedded |
| Production permission | Twenty independent synthetic authorization checks, including invalid-binding rejection before scientific imports or run artifacts |
| Bounded GPU fixture | Exactly128 TRAIN molecules, two batches,12 discarded optimizer steps; all arms pass |
| Numerical replay | Model max1.1921e-7; Adam max2.9802e-8; loss difference0; RNG/order exact |
| F gradients | Identity mixing gradient norm~2.7624e-4; upstream gate gradient~1.46e-5 after first step |
| Penalty scale | Weighted decorrelation/base gradient norm ratio~1.1896e-4; no coefficient tuning |

The early-row fixture peak allocation was0.62–0.99GB and the wrapper completed in29.74s. Those rows are small molecules; this is not a worst-case full-TRAIN resource estimate. Historical original-MTO throughput suggests roughly2.5–3h for60 epochs per arm, with four concurrent admitted GPUs and25% operational headroom pending first-epoch measurement.

## Preserved failures and explicit corrections

The first CPU export fixture converted the whole model FP32→FP64→FP32. This changed mto.n_ref from its native FP64 dtype to FP32, with unchanged value, and the strict buffer fingerprint correctly rejected it. The repaired fixture constructs a native-dtype model and transfers only hand-set adapter tensors. Original source/log and diagnosis are retained.

The first GPU fixture failed full-forward bitwise equality before any optimizer update. A separate zero-update diagnostic found repeated identical model forwards also vary, while F/CG/decoder on identical raw M are exact. Root explicitly approved applying atol2e-6/rtol1e-5 to independent GPU full-forward/export comparisons, retaining exact CPU/shared-M checks and original update/loss criteria. This was an acceptance-rule amendment after diagnosis; no kernel was conclusively isolated. All failed and diagnostic records remain intact.

After successful GPU checks, final training.py adds only parent-directory fsync to two file-renaming helpers. The executed source, exact old→snapshot→new mapping and independent AST/text-diff continuity review are preserved. No additional optimizer updates or I/O runtime tests were performed for this narrow change. The final source and executed mathematical path are distinguished explicitly in the frozen provenance.

## Production and recovery contract

The reviewed future recipe is fixed60 epochs, seed/order11, Adam AMSGrad LR.001, batch64, WD0, clip5, FP32/no AMP/TF32, original LE+Ls plus fixed arm penalty. Every arm has1881 batches/epoch and112860 total updates. All66860 validation states are scored after each completed checkpoint, including epoch0. Minimum pooled raw-f SSE selects the earliest tied checkpoint. Per-state, TRAIN-defined bright-tail/false-bright and energy metrics accompany pooled results.

last.pt is the atomic completed-epoch commit and automatic restart point, containing model/optimizer/RNG/order/history and immutable selected-version hashes. best.pt preserves the selected model/optimizer/RNG snapshot. geometry_best.pt will be one self-contained geometry-only inference checkpoint. Mid-epoch work is replayed from the preceding committed epoch; no disposable preflight state enters production. All weights, optimizer states and raw/prediction/index arrays stay server-only.

Production launch is still blocked. Root must issue a distinct exact authorization only after final independent closure review and verified D-first publication. PRODUCTION_AUTHORIZATION_TEMPLATE.json is deliberately unauthorized. Future command:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation
/usr/bin/python3 launch.py --authorization /absolute/path/PRODUCTION_EXECUTION_AUTHORIZATION.json
```

The launcher admits control/F/decor/both sequentially to physical GPUs1/2/4/6, with a separate retained shared lock per worker. Each child receives its physical UUID before interpreter imports, two threads and unbuffered logs, and is registered by exact process identity. Existing four-hour monitoring remains in place. A later admission failure leaves already admitted owned workers intact; no unrelated job is signaled.

## Limits and next decision

No v2 accuracy result exists yet. The new TEST partition contains5989 historical TRAIN,361 historical validation and336 historical TEST rows; it is a new disjoint internal partition of exposed data, not external fresh evidence. Old model scores cannot serve as matched v2 results. The full-training trajectory is not guaranteed bitwise deterministic on CUDA.

At60 epochs, a noncontrol arm needs at least .003 validation R² above its contemporaneous control for the predeclared common continuation to100 to be considered. Otherwise close this bounded pilot. Any continuation, independent-seed confirmation or final TEST scoring requires its own reviewed decision. No evidence currently guarantees that this change alone reaches R² .60.
