# Round09 preparation report — original MTO coupled Adam decay

The implementation and bounded CPU/GPU checks passed. This is engineering evidence only; no validation accuracy or regularization benefit was measured. Final independent preparation review/root acceptance/publication and separate production authority remain required before60-epoch fitting.

## Exact comparison implemented

Both arms use the unchanged original MTO/PSD decoder and LE+Ls, fresh seed/order11, one identical ordered135-tensor optimizer group containing1,552,092 elements. The complete group matches the unwrapped original trainable name/shape/dtype/numel list;11 dormant right-F/transport tensors are frozen. No bias/norm/embedding/radial/offset exemptions. The only optimizer option difference is coupled AdamAMSGrad weight_decay0 versus1e-4. Decay is added after task clipping and enters both moments/AMSGrad, with default dispatch and grad=None skip. It is not AdamW or a pre-clip norm penalty. No coefficient, group, LR, clip, objective, tolerance or state-initialization change occurred.

PARAMETER_ROSTER.json SHA290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943 was frozen before any optimizer calls. Both arms rebuild common base SHA231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2/full SHAf9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3. No trained checkpoint or disposable fixture state may initialize production.

## Completed checks

| Check | Actual result |
|---|---|
| Production authorization |22 independent stdlib-only synthetic binding checks passed, including denial before scientific imports/artifacts. No production authority created.|
| CPU optimizer arithmetic |Exactly5 toy Adam step calls; manual coupled-afterclip moments/bias correction/AMSGrad retained maximum, None-versus-zero and preclip/AdamW distinctions passed. Maximum FP64 difference2.22e-16.|
| CPU full model |Three fresh constructors, zero model updates. Exact roster/base/full hashes, equal initial task loss/gradients, target masks/zeros, symmetry and one-file synthetic export/access checks passed.|
| TRAIN GPU fixture |Exactly6 discarded updates: update1/update2/replayedupdate2 per arm, first128TRAIN rows/two64batches. All135 original gradients present; frozen parameters/buffers unchanged. No VAL/TEST numeric rows.|
| Resume |RNG and order exact; largest model difference1.1920928955078125e-7, Adam difference1.4901161193847656e-8, task-loss replay difference0. Fixed atol/rtol held.|
| Geometry export |E/A passed fixed numerical bounds against each model's own export, with strict one-file load/forward access. Native-f errors were reported from the same outputs; maximum2.008828504582172e-9. No separate f acceptance threshold or production inference.|

CPU_PREFLIGHT SHA dfca79770fa451a64e08bd936b80294301829d527a10e60a48ccfe0e98d854c5; independent CPU reviewc2147eb3. GPU_PREFLIGHT SHA a32a7fe8759e1ca171c6dee2efca91374c5f1e749e5a95235cacb6682add77ac. Original mathematical technical review5c4b82c0 remained unchanged; resource retry reviewa62e9a8e separately binds the execution amendment.

All six GPU updates clipped task gradients. In the coupled arm, detached decay-term norms were.0807565 then.0806883, about.01615/.01614 of the clipped task-gradient norm. Reconstructed effective-gradient norms were about5.00045–5.00046, illustrating that adding decay after clipping does not preserve the task clip bound. These are detached FP64 diagnostics, not a claim of bitwise identity to an optimizer kernel's intermediate arithmetic. They do not calibrate the coefficient or establish a generalization mechanism. Radial beta minimum absolute value remained finite (2 then about1.999); no clamp or exemption was introduced.

## Preserved admission failure and explicit retry

The original GPU1 wrapperPID1863309 failed the fixed memory<1000MiB admission check before any scientific child, registration, target decode or update. The original ops/gpu_preflight_attempt remains empty. Nine exact source/review/log/XML copies are preserved in admission01_preserved/MANIFEST.json SHAab9c820e719b241021879d786a5b78f04d95f103fe0e941344ea7e4592529e65; independent admission reviewb1b9e401 verified this boundary. Unrelated processes were untouched. Two earlier reviewer metadata checks found missing identical server helper copies; exact copies were restored and both failed binding records preserved, with no numerical rerun.

Root resource decision673bcc503705fe6063e14f02d3573738d82a12907f18e730d41d760c7ae277e8 permitted one distinct retry. Fresh independent inventory selected physicalGPU4 first in ordered4,6,2,1. The retry wrapper re-admitted GPU4 under its exclusive shared lock, prebound UUID GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184 before scientific imports, registered childPID1929399/start1289211519, and released the original registration barrier. The unchanged gpu_preflight.py then ran once. No CPU/roster repeat, source/math change or automatic fallback occurred.

Actual terminal: ops/gpu_preflight_attempt_retry01/COMPLETE.json, exit0/registration0; wrapper elapsed18.40s, fixture aggregate13.65s. Three executed updates plus associated checks took3.09s for zero_decay and2.25s for coupled_l2; peak allocated CUDA bytes664,061,952 and668,020,224 respectively. This small early-TRAIN fixture does not represent worst-case full training batches or predict complete training time. Future production admission remains independent.

## Reproducibility and remaining limits

PROTOCOL.md records the accepted math/settings; RUNNER_HANDOFF.md supplies the exact blocked command, authorization schema, GPU1/2 production plan and completed-epoch recovery rules. Original scientific files/review remain intact; RESOURCE_RETRY_CONTINUITY and separate launch/terminal records identify the resource-only change. The final metadata freeze binds source, installed optimizer text, receipts, roster, split and60 prescribed TRAIN orders. It does not decode targets or construct a model.

The proposed production budget remains60epochs/112860updates per arm, all66860 native validation labels includingzeros, earliest minimum SSE with epoch0 eligible, and candidate+.003 above BOTH contemporaneous control and retained.44716940136585204. No automatic seeds, extension, alternate decay, new exclusions or TEST scoring. TRAIN diagnostics are pre-update trajectory aggregates, not fixed-checkpoint TRAIN evaluations. Repeated original-control variability and late validation deterioration do not prove a cause or predict a decay gain.

All technical model/optimizer checkpoints stay server-only and are forbidden as initialization. No production runs or production authorization exist. Sealed TEST remains unopened. The0.60 target remains unmet; neither CPU arithmetic nor the six-update fixture establishes empirical progress toward it.
