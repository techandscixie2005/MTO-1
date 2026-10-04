# Proposed engineering amendment — awaiting root decision

This document is a proposal, not execution authority. Preserve the original TECHNICAL_PREFLIGHT_PLAN.md, accepted protocol and failed source/log snapshots. No retry update preflight is authorized by this document alone.

The first GPU preflight stopped at its full-forward bitwise identity assertion before any optimizer step. Snapshot0f9fd724861c905582524f071c310d73d0e161abcef75c7b9c15aaad63e94efe preserves its exact source/review/log. The only private preflight artifact is the disposable marker; no after-update checkpoint exists.

Reviewed diagnostic4221233ca22bb3cf0fdfb2091a20ae500d72c4ddf5fc4aec0e44277307b3d92c used only z/pos/edge for the first two already authorized TRAIN geometries, with zero target decoding or optimizer updates. Repeated forwards of the original model itself varied. Across repeated original/control/F forwards, maximum E difference was9.536743e-7, A9.313226e-9 and native-f3.833712e-9. Raw M also varied across repeated forwards. Cross-model differences were within observed repeat variability. On identical raw M, original and wrapper CG/decoder operations and identity F were bitwise equal. The evidence is consistent with CUDA execution/reduction variability rather than an adapter functional discrepancy, but it does not isolate a particular kernel or prove whole-trajectory determinism.

Proposed checks:

1. Retain exact inherited/full initialization tensor hashes and CPU full-forward identity.
2. Retain GPU bitwise F and CG identity on the same raw M, removing the upstream repeated-backbone confound.
3. For independent end-to-end GPU full-forward/export comparisons, use atol2e-6/rtol1e-5 and report E/A/native-f absolute differences and bitwise flags. These numeric values were already fixed for model/optimizer replay, but **their application to this full-forward/export acceptance rule is new after diagnosis**. Do not describe that rule as unchanged.
4. Keep model/optimizer next-update and loss tolerances unchanged. Keep exact RNG/order checks, the same first128 TRAIN rows/two batches, all scientific settings and total12 discarded optimizer steps unchanged. No validation/test access or accuracy-based change.
5. Preserve the original failed directory and use distinct repair01 private/ops paths, owned ID and exact re-reviewed source. No production state is initialized from any technical artifact.

If root accepts, a new bound technical source review and fresh GPU admission precede the first actual update preflight. If it rejects, keep production blocked and investigate a different exact comparison without changing model settings.
