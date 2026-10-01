# Independent Round07 preparation review

**PASS for preparation only.** No blocking findings remain. The reviewed source closure is `68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203` (185 files). Production is not authorized.

## What was reviewed

- The original, common scalar and shared tensor arms implement the accepted equations and exact 177-parameter gate. Gate initialization preserves the original function without an identity shortcut; the original gate is frozen and bypassed in its objective graph. The older right-F schema is dormant in every arm.
- All ten predicted state slots supply the fixed nine inputs independently of labels or masks. Original LE+Ls, TRAIN-only normalization, predicted-energy gradients, fresh base initialization, batch/order/optimizer/budget and native raw-f checkpoint selection are preserved. The scalar and tensor controls match active gate capacity, inputs and perturbation norm; their possible output effects differ.
- PSD, O(3), factor gauge and qualified real equal-energy block-sum properties are supported by algebra and fixed synthetic fixtures. These do not identify electronic transition tensors, polarization, X/Y, densities or NTOs. Fixed-input rank/zero and common-sign limitations remain.
- The selected-row reader is unchanged; the technical fixture decoded only the first128 v2 TRAIN rows. Validation/TEST numeric rows were zero. The v2 split remains a historically exposed repartition with audited identity-rule disjointness, requiring fresh initialization.
- A standalone export embeds the full model, one transform, configuration, TRAIN statistics, exact readout contract and buffer fingerprints. CPU access guards and the trained disposable GPU export checks passed. Inference needs geometry and one checkpoint, without QC labels, training readers, source checkpoints or caches.
- Committed last.pt retains model/Adam/RNG/order/history and immutable selected prediction/checkpoint references with durable writes. Incomplete epochs replay from their last completed commit; completed arms refuse relaunch. Exact execution authorization, source/review/publication bindings and fresh GPU health/UUID/lock admission precede future production.

## Exercised evidence

Science ran the CPU stage once with zero updates, then the reviewed GPU fixture once: update1, update2 and restored update2 for each arm, exactly nine discarded updates on128 TRAIN molecules. All numerical inputs and settings remained unchanged after execution.

Initial CPU/module identity, full base/schema/gate hashes and synthetic export parity were exact. Algebra, mask, analytical-gradient, reflection/permutation/translation, factor-gauge and limited degeneracy tests met their fixed criteria. Active gates had nonzero initial gradients; original gate gradients stayed zero. Modified trace-to-energy feedback was zero initially and approximately5.75e-9 after one update. All nine steps used the preset clipping at5; no coefficient or learning-rate adjustment followed.

Saved/restored RNG and fixture order were exact. Replayed loss difference was0; maximum model/Adam differences were1.1921e-7/1.4901e-8, within the prescribed combined absolute/relative tolerances. Full GPU and own-export parity were numerical, not bitwise. CPU constructor isolation is distinct from CUDA setup and explicit resume restoration: torch.manual_seed may reset initialized CUDA generators even inside a CPU-only fork.

The reviewer independently exercised22 temporary-file authorization checks and two stdlib-only metadata/hash checkers. The latter verified ownership/admission/normal exit, update logs, per-field access ledger, six opaque private-checkpoint hashes, all185 source bytes, all134 inherited dependencies,60 order hashes, proposal/settings/init/statistics equivalence and receipt closure. They did not decode model tensors or dataset arrays, construct a model, repeat inference, or add updates. Source inspection covers the unexecuted production launcher, evaluator, recovery and selection paths; a full production trajectory has not been tested.

## Interpretation and remaining gates

No empirical accuracy gain or seed robustness is established. The small fixture had q near1/3, so it does not show that learning will exploit anisotropic context. The existing decoder uses nonzero small random tensor-gate initialization; source review found no exact zero-anisotropy gate-initialization trap. Tiny early molecules are not a worst-case resource bound. Trace supervision leaves tensor orientation unidentified, while single-seed CUDA variability and bounded corrections remain material limitations.

The unchanged tensor allocation rule requires +.003 native pooled R² over both contemporaneous controls and retained reference .44716940136585204. Better controls remain reportable; passing only permits a later root decision on independent seeds. No averages, validation-driven setting changes, automatic extensions or TEST scoring are authorized. The .60 target remains unmet.

Remaining steps are root preparation acceptance, D-first lightweight archive and remotely verified publication, then a distinct exact root production authorization. Preserve the185-file seal and completed numerical stages. Keep every weight, optimizer/gate tensor, dataset, prediction, private identity/split array and cache server-only; private hashes are provenance references and must not be expanded into archive entries.
