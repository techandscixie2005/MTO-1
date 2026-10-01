# Round07 preparation: shared PSD context with two controls

Implementation and bounded engineering checks are complete. The 185-file source closure is sealed as `68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203`. Independent final review, root acceptance and publication are separate receipts. No production fit or validation/TEST scoring occurred during preparation. No empirical accuracy gain is claimed.

## Concrete comparison

Three fresh models retain the original PSD MTO decoder and original LE+Ls. `original` bypasses the new readout, `scalar` applies a common molecular strength rescale, and `tensor` applies a shared congruence to every state tensor. Both modified arms train the same 177-parameter gate on the same nine invariant inputs. All arms retain the same checkpoint schema; the original arm freezes the gate. The older 4,016-parameter right-F schema is disabled and frozen in all three.

With R equal to the sum of ten incoming A tensors, B=(R+epsilon I/3)/(tr(R)+epsilon), q=||B||F/sqrt(3), and b=.25 tanh(g), the scalar output is (1+bq)^2 A and the tensor output is (I+bB)A(I+bB)^T. E is unchanged in the forward transform. Its predicted values remain differentiable gate inputs, so trace loss can reach the energy branch after the gate moves. Epsilon, scales, features and bounds were fixed before any check.

The existing decoder already produces coherent C C^T outputs. This study tests shared context from latent tensor overlaps, not missing PSD expressivity or a demonstrated electronic-response mechanism. Scalar and tensor match gate capacity, inputs and Frobenius perturbation norm; they do not match every possible output effect. Tensor directions remain unidentified by trace-only supervision.

The proposed production budget remains three matched 60-epoch runs, fresh seed/order11, original Adam AMSGrad LR .001, batch64, clip5, weight decay0, FP32 without AMP/TF32, two CPU threads. No numerical setting changed after inspecting preflight outcomes. No trained checkpoint or disposable fixture may initialize production.

## Engineering evidence

| Check | Verified result |
|---|---|
| CPU transform identity | Exact on the same incoming E/A; all initial full-model outputs match the original |
| Symmetry and gauge fixtures | Proper/improper O(3), atom permutation/translation, factor gauge, isotropy and the limited real equal-energy block-sum property pass |
| Bounds and suppression | PSD/rank/bounds, zero/tiny/bright inputs and hand-set negative-b suppression pass |
| Gradients | Analytic scalar/tensor derivatives, final-layer factor .25, target-mask sentinels and differentiable energy inputs pass fixed criteria |
| Authorization | 22 independent synthetic checks include valid binding and invalid-binding rejection before scientific imports/artifacts |
| Fresh initialization | Identical base, inherited disabled-F state, full schema and gate hashes across all three arms |
| Actual TRAIN scope | First128 rows; two batches64; update1, update2 and replayed update2 per arm: exactly9 discarded updates |
| Resume | Exact saved/restored RNG and order; maximum model difference1.1921e-7, Adam difference1.4901e-8, replayed loss difference0 |
| Geometry export | One-checkpoint CPU load/forward access guard and each trained disposable GPU export pass |
| Data boundary | Selected-row decoder ledger records128 TRAIN rows per requested field; zero validation/TEST numeric rows |
| Resource ownership | Fresh physicalGPU1 admission, shared lock, UUID before imports, registered child identity and normal exit0 |

Active scalar/tensor gate gradients were nonzero at identity (norm about .00760); the original gate remained frozen. Initial trace-to-E feedback was zero, then became about5.75e-9 after one update in the modified arms. All nine steps clipped at the fixed norm5. This demonstrates that the reviewed graph is connected; it does not estimate optimization quality, validation performance or physical interpretability.

Same-input module identity and CPU export comparisons are exact. Full GPU comparisons meet the predeclared combined absolute/relative tolerances rather than bitwise equality. CPU constructor RNG isolation and exact fixture resume RNG are distinct checks: `torch.manual_seed` can reset initialized CUDA generators, while production setup after construction and explicit resume restoration provide the reviewed runtime behavior. Equal starting tensors and TRAIN orders do not guarantee identical full CUDA trajectories.

The owned wrapper took24.63s, with three-update sections about3.19s/2.48s/2.49s. Peak allocated memory was at most668,203,520 bytes. These early TRAIN rows are not a worst-case memory or full-training timing bound. In this short fixture q was close to1/3, so it does not show that training will exploit anisotropic context; separate synthetic fixtures exercise that algebra. The production estimate remains roughly8 GPU-hours, with25% scheduling headroom, based on prior complete runs rather than extrapolating these tiny batches.

CPU and GPU numerical stages each ran once and passed without a tolerance amendment or retry. Historical preparation failures and fixes remain bound through the inherited source closure; they are not new Round07 failures. Completed numerical stages must not be rerun during closeout.

## Selection, limits and future allocation

Each arm will select the earliest minimum pooled native raw-f validation SSE among epochs0–60, using all66,860 labels. Report selected and fixed60 scores, all states, energy, TRAIN-defined true bright tails at .0549/.2406, false-bright bins and error concentration. TRAIN diagnostics are pre-update trajectory aggregates, not final-checkpoint TRAIN evaluations.

Tensor allocation requires at least +.003 R² over both contemporaneous selected controls and the retained v2 reference .44716940136585204; thus the reference threshold is .45016940136585204. Passing permits a later root decision on paired fresh seeds, not automatic continuation. A better scalar or original checkpoint remains a descriptive reference without proving tensor context or triggering automatic allocation. No seed predictions are averaged.

The v2 TEST stays sealed. It is an internal repartition with historical exposure and disjointness under audited identity rules, not external fresh confirmation. No empirical gain, seed robustness, oscillator sum rule, polarization recovery, TDDFT amplitude or NTO reconstruction follows from these checks. At fixed incoming A the transform preserves rank and cannot revive a zero tensor; all strengths change with the sign of a common b. The trainable base can still change its inputs. Single-seed variability, latent orientations, bounded correction strength and early overfitting remain uncertainties. The .60 research goal is unachieved.

## Execution boundary and artifacts

The exact currently blocked command and authorization schema are in `PROTOCOL.md` section9. The future launcher admits original/scalar/tensor on physical GPUs1/2/4 sequentially under separate locks, then runs them concurrently. UUIDs are prebound before Python imports; there is no automatic device fallback or blind restart. Root acceptance, a D-first inspected publication receipt and a separate exact production authorization must all be present. The false template grants no authority.

`last.pt` is the committed resume entry with optimizer/RNG/order/history. `best.pt` is the selected model/optimizer snapshot; `geometry_best.pt` is the standalone model/config/statistics/transform-contract export. All fixture states stay server-only and are excluded from initialization and publication. Archives contain code, contracts, logs, aggregate results and opaque provenance hashes only; no weights, gate tensors, optimizer state, datasets, predictions, identities or caches.

### Bound records

- Final implemented protocol: `ee22c436332e753de1636e3db5e6587a628f6e7925965748cd1a626abf710bda`.
- Root preparation decision: `637a3cd3006d3f5b84f8ca5314b6213b0bc3017dffeeafb7ed51c2a4c594c60b`.
- CPU preflight: `b5f7da4fad6f27f6b93e204d561594f9040946c46a3d6c5e48fbeb70f59ef777`; independent review: `27cb7fc4025de1595399549bcf359cb5da83a97f3f035b4bd1a379e5ed881a8c`.
- GPU technical source review: `5dff6ce331d9cb7cb5f17d54e0d6968b9317e2a4653cc1c2c8821d774a489f82`.
- GPU preflight: `24e891e92c8c670361f406363153c5a39719106416108bcfc9b56813598296b6`; owned terminal: `e19ac28e45c70c048d70e7c0e4607e9ce77c0259d4d1654bf6c0429bdb40e21f`.
- Independent GPU review: `e0a882ff6d44c9f221dbda56d46510a99dccf0105e2d3efa9afb1ff60b9a1b5b`.
- Independent authorization checks: `e9b328e6dad04acf407df931444b801db2ea5f6f6a5ad5be07f817f2bdc75fed`.

Final independent preparation review and publication must bind this exact closure. This report is a supplemental lightweight record; it does not amend the sealed implementation or confer execution authority.
