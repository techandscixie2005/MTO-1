# Round06 preparation: original PSD, two objectives

The implementation and bounded engineering checks are complete. The 134-file source closure is frozen as `4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b`. Final independent review and preparation publication are recorded in separate bound receipts. No 60-epoch fit or validation/TEST scoring has occurred during preparation, and preparation grants no production authority.

## Concrete comparison

Two fresh, matched original-PSD MTO models compare LE+Ls against LE+Lf. Lf is printed raw-f MSE divided by the frozen v2 TRAIN population variance .0025089892829484074. Native f=2 E_eV tr(A)/(3×27.211386245988), with gradients through predicted energy and A. Both arms retain coefficient1, seed/order11, original Adam AMSGrad lr.001, batch64, clip5, weight decay0, FP32/noAMP/noTF32 and60 epochs. There is no adapter, decorrelation loss, QC-label input, calibration, averaging or checkpoint warmstart.

The question is narrower than historical raw-f trials: the .373493 scratch study changed to a direct-f head, whereas original-PSD raw-f trials used trained warmstarts or a frozen backbone. Prior failures remain negative evidence. The current comparison changes only the objective on the same newly initialized original model.

## Engineering evidence

| Check | Result |
|---|---|
| CPU original control | Bitwise equal to original base loss |
| Synthetic E/A/C derivatives | Maximum absolute discrepancy9.27e-7, within fixed bounds |
| Invalid/zero labels | NaN-invalid entries masked before nonlinear arithmetic; finite valid and zero invalid gradients; printed zeros retained |
| Authorization |22 reviewer-owned synthetic checks, including valid binding and rejection before scientific imports/artifacts |
| Fresh initialization | Both arms match frozen original-base and full-schema tensor hashes |
| Actual TRAIN technical scope | First128 rows, two batches64, update1/update2/replayed update2 per arm: exactly6 discarded updates |
| Resume | RNG/order exact; model max1.19e-7, Adam1.49e-8, loss0 |
| One-checkpoint inference | Construction and geometry forward succeeded with external data/checkpoint access blocked |
| Data boundary |12 selected-decoder ledger entries; each has exactly128 TRAIN rows, zero validation/TEST numeric rows |
| Owned resource execution | PhysicalGPU1 UUID bound before imports; fresh health/lock admission, registration and normal child exit0 |

The fixed initial minibatch's raw-f intensity gradient norm was6.7829 times its trace-intensity gradient norm. Both E and PSD-head gradients were live; all6 steps clipped at the prescribed norm5. This demonstrates unequal gradient scale despite dimensionless losses. No weight, LR, threshold or budget changed in response. It does not estimate validation accuracy.

The wrapper took19.01s; three-update sections took2.79s/2.11s with peak allocated memory below621MB. Early-row fixture timing/memory is not a full-dataset bound. Prior full Round05 timing suggests roughly2.7h wall time for two concurrent60-epoch arms,5.4GPUh plus overhead; allow25% scheduling margin.

The first wrapper invocation rejected an unmirrored, already-reviewed ENVIRONMENT.json before any resource admission, child, data access or update. History mirrored the exact pinned bytes and rechecked all116 technical bindings. The subsequent invocation was the only actual GPU fixture. This provenance failure is preserved in `ops/GPU_PRELAUNCH_GUARD_FAILURE.md`; no scientific change or tolerance relaxation occurred.

## Selection, interpretation and remaining limits

Earliest minimum pooled validation raw-f SSE selects from epochs0–60, all66,860 labels. Report selected and fixed60 results with per-state, energy, true bright tails and false-bright bins. Candidate allocation requires at least+.003 R² above both the contemporaneous selected control and the retained v2 reference .44716940136585204. Failure closes this fixed comparison unless a later explicit decision supplies a new rationale. Passing does not itself authorize new seeds or extension.

TRAIN losses are pre-update trajectory aggregates. TRAIN `total` is the chosen objective; TRAIN `base` and validation `base_objective` remain LE+Ls. FP32 normalized training Lf differs slightly from the FP64 raw-f MSE diagnostic. Equal initial tensors/orders do not establish bitwise CUDA trajectories. The engineering receipt checks tight numerical replay, not deterministic whole training.

The v2 TEST partition remains sealed. It is an internal repartition with historical exposure, not external fresh confirmation; no old trained weights can initialize this round. No empirical gain is claimed. Single-seed uncertainty, outlier sensitivity, coupled E/strength compensation and the unchanged PSD feature interpretation remain limitations. The0.60 research goal is unachieved.

## Execution and artifacts

Exact commands, authorization schema, GPU1/2 admission plan, no-blind-retry rules and inference recipe are in `PROTOCOL.md`. All current production commands are blocked. Root preparation acceptance, D-first byte-reviewed publication and a separate exact `PRODUCTION_EXECUTION_AUTHORIZATION.json` are required before launch. The false template is intentionally non-executable.

`last.pt` will be the committed resume entry with optimizer/RNG/order and completed history. `best.pt` is the selected snapshot; `geometry_best.pt` is the standalone model/config/statistics export. Disposable preflight states never initialize production and stay server-only. All weights, optimizer tensors, datasets, prediction arrays, IDs and caches are excluded from archives. Lightweight code, contracts, logs, aggregate receipts, reviewed failures and exact dependency hashes are preserved.

Current important receipt hashes:

- Proposal review: a1617eaea6574b36f5e39fa9b5bb738c7a2ba91083c373217e2c51d25b992225.
- CPU preflight:02c268678eb4b6293258ec99020d4f7f16c6d9f3244ccf122ea4ebae7a6e8bd9; independent evidence review:9cb76884bb5bd189ea1e9d090ce9db83b396654e6e497c7932c5f776b31d088f.
- Technical source review:fbf76a414d89a459cf5fa3c956544a698b1f564c856622c107fa99b6dc592605.
- GPU preflight:cdb0520de4b80a3618114ad02ed61a34a071cf02701f0dbfd62168efcf6d2aad; owned normal terminal:ecfe67939ff7c2cf9b5a7b6d49db259e97e6d97af46adba8c4f07f76f890bb0d.
- Independent GPU evidence review:fc39d7e2b915717e074b4eb99c1971a447f72d13b4bfe642eae583633eedfee0; source lineage:454ee6189e2435b36dabc005c3fa4d6c77252fff12778c7960794b46feef68e8.
- Independent authorization checks:60aa32905289a8a01fd612fad431fc0488c57f464b5854a06b14f172a8595649.
