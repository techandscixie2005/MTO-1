# Bounded engineering preflight plan

Status: fixed before execution. These checks are permitted only after the repaired QM9S v2 split has completed its independent saved-array verification and is frozen. No production60epoch fit is authorized. The proposal/root preparation decision controls scientific settings.

## Access and isolation

- New TRAIN statistics may be computed once over all new TRAIN rows with the streamed selected-row decoder, separately from the small model fixture. Record actual archive/member names, global-index content hashes and row counts for every decode. Header parsing and skipped compressed bytes are not numeric target decoding; no entire numeric target array may be loaded first.
- The model fixture is exactly the first128 rows of the new TRAIN in its frozen ascending original-row order. Save its index hash and derive its ID hash from metadata. No label-based choice or replacement. Two consecutive minibatches of64 are the only real-molecule optimizer inputs.
- Model parity/transform/export probes use at most the first two of these TRAIN molecules and two separately defined synthetic molecular geometries. No validation/test target or geometry fixture is needed. A separate evaluator reader has access only to new validation indices after production authorization; no test-target API is implemented.
- All technical models, optimizer states and any serialized replay files live in an isolated preflight directory and are marked disposable/non-production. The production launcher cannot load this directory. It independently reconstructs the seed11 base and verifies its frozen fresh-initialization hash before any updates.

## Exact model-update budget

For each of the four prescribed arms, perform one update on the first64 TRAIN molecules, serialize optimizer/model/RNG/order state, then one uninterrupted update on the next64. Reconstruct from the serialized first-step state and repeat only the second update for numerical resume comparison. This is **three executed optimizer steps per arm, twelve total**, two unique minibatches per arm,128 unique TRAIN molecules. No epochs, checkpoint selection or accuracy comparison occurs.

Identity/loss/gradient diagnostics may use forward/backward calculations on the same first minibatch without optimizer updates. Finite- and live-gradient checks must distinguish the zero-initialized final adapter mixing from initially zero upstream gate gradients. Inspect gate activation after the first discarded update; do not choose a different fixture to make a diagnostic pass. Raw-M penalty magnitudes and lambda-weighted gradient ratios are diagnostic, not a coefficient sweep.

The resource reservation is at most30minutes on one admitted healthy A800 for these checks, expected to take only a few minutes; this is a preflight planning allowance, not a permission to kill an unidentified process. CPU synthetic/decoder/export checks need no GPU. Prebind the physical UUID before interpreter imports, acquire the shared GPU lock, and save owned PID/start/boot/UUID/cwd/argv and admission evidence.

## Fixed tolerances and claims

- Initial base tensors and shared data-order arrays: exact byte hashes. The adapter-disabled and identity-enabled wrapper outputs must be bitwise equal to the equivalent freshly initialized original forward on the same device and batch when they execute the same operations; an unexpected mismatch stops for diagnosis.
- Standalone FP64 adapter rotation/reflection error: <1e-10; raw-feature penalty invariance error <1e-12, matching the existing mathematics checks.
- Full FP32 model O(3): energy atol2e-5/rtol2e-4, A atol5e-5/rtol5e-4, inherited from the established e3nn/backbone check. Report native-f differences explicitly. Atom permutation/translation use the same model-output tolerances with remapped edges.
- Serialized next-update model and optimizer floating tensors: atol2e-6/rtol1e-5, matching the earlier source preflight. Integer fields, step/order/cursor hashes and shapes/dtypes are exact. Report observed maximum model and optimizer differences separately; loss replay atol1e-6/rtol1e-5. Restore optimizer tensors through CPU map_location so Adam step counters retain the intended device behavior.
- These checks establish bounded numerical next-step replay, not guaranteed bitwise CUDA training trajectories. Scatter kernels may be nondeterministic. No post-outcome tolerance relaxation or replacement of failed records.
- Composite checkpoint synthetic/same-batch load-forward: exact state/config/statistics/buffer hashes and same-device predictions; no external data, cache, original weights or statistics file may be read after checkpoint loading. All constructor inputs come from the one checkpoint plus published code.

## Required negative checks

Synthetic containers with forbidden target sentinels must prove rejection before numeric decoding for: TEST rows, unknown fields, duplicate/out-of-order/out-of-range selections, mismatched source/split hashes, and direct full-array target loading. Access receipts must be based on the actual decoder calls, not a manually asserted list. Masks are applied before arithmetic; no valid zero label is dropped.

Production entrypoints must reject missing/mismatched execution authorization, source review, split verification or publication receipts before target reads or run artifacts. Completed fits refuse rerun. Resume verifies arm/config/split/source/initialization hashes and restores all state. Epoch interruption replays from the preceding atomic complete-epoch checkpoint; incomplete progress is not treated as a new independent experiment.

Any preflight failure is preserved and diagnosed as an engineering issue. Successful technical checks are not evidence of empirical accuracy or a promotion. Final source/preflight independent review, archive-first publication and a new bound execution decision remain required for the60epoch pilot.
