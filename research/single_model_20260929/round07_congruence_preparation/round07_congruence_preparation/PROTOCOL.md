# Round07 implemented preparation contract

Root preparation decision `637a3cd3006d3f5b84f8ca5314b6213b0bc3017dffeeafb7ed51c2a4c594c60b` authorized implementation, fixed synthetic checks and source-reviewed nine discarded TRAIN updates. Those stages are now complete and independently reviewed. Production remains blocked pending final preparation review, root acceptance, D-first publication and a separate exact execution authorization. `PROTOCOL_PROPOSAL.md` preserves the original documents-only snapshot unchanged; this file states the implemented contract.

# A shared transform of latent PSD state tensors

Round06 is complete and published at `68520927333b27c48381401c8516a6709b79c40e`; completion receipt SHA `80bb3a01c2f65e08c5b7718d5bf60a3736af4572fe66f310a9dcdb46974cb889`. Completed rounds remain unchanged. This preparation used synthetic CPU fixtures and exactly nine discarded TRAIN updates; it did not run a production fit or decode validation/TEST numeric targets.

## 1. Question and three required arms

Does conditioning each state's PSD output on the aggregate orientation of all predicted state tensors improve pooled raw-f accuracy beyond both the original predictor and a common molecular scalar rescale?

| Arm | Output transform | Active added parameters |
|---|---|---:|
| original | A'_k = A_k | 0 |
| scalar | A'_k = (1+bq)^2 A_k | 177 |
| tensor | A'_k = (I+bB) A_k (I+bB)^T | 177 |

All arms use the original PSD MTO16/query32/router128/head128 with ten states, all original parameters trainable, original LE+Ls, and the same fresh initialization. The reviewed disabled-F constructor is retained: F is disabled and its 4,016 schema parameters frozen; raw-M penalty is zero. Instantiate the same new gate in every arm, freezing and bypassing it for `original`. This gives a common checkpoint schema; it does not make the original arm actively capacity matched. The scalar and tensor arms have the same active gate parameters and inputs. No raw-f loss, calibration, new QC label, dropout, weight decay, learned knot, cap, eigensolver or prediction averaging is added.

The current decoder already forms C=beta I/sqrt(3)+Q from coherent channel interactions and A=C C^T. It already spans PSD tensors. The proposed contribution is shared cross-state tensor context, not missing PSD expressivity or absent coherence. Tensor directions are latent under trace-only supervision, not identified physical transition polarizations.

## 2. Exact differentiable transform, units and gate

For every molecule, use all ten **untransformed** predicted A_k and E_k. Define E_H,k=E_k/27.211386245988 with E in eV, s_k=tr(A_k), R=sum_k A_k, and S=tr(R). Strength is in atomic-unit squared electric dipole. Fixed constants are epsilon=1e-6 in those units and s0=1 in the same units:

    B = (R + epsilon I/3)/(S + epsilon)
    q = ||B||_F/sqrt(3)
    b = 0.25 tanh(g(x))

For symmetric B, q is exactly sqrt(tr(B^2)/3), the existing deferred design. The Frobenius form explicitly defines the matching norm and avoids an unnecessary diagonal matrix-product reduction. It is an algebraic specification, not a data-driven change. Do not symmetrize, clip, floor or repair outputs silently.

The fixed nine gate inputs, in this exact order, are:

1. log1p(N_H), log1p(N_C), log1p(N_N), log1p(N_O), log1p(N_F), from the molecule's real atoms (padding, if present, is not an atom; unsupported elements fail).
2. mean_k E_H,k over all ten slots.
3. log1p(S/s0).
4. log1p(sum_k E_H,k*s_k/s0).
5. q.

There is no learned input normalization, LayerNorm, batch statistic or TRAIN-fitted feature standardization. The fixed unit scales and epsilon are not selected from validation or gate-output ranges. All features are computed independently per molecule; no batch mixing. There is one shared gate for all states, g=Linear(9,16)-SiLU-Linear(16,1), totaling (9*16+16)+(16+1)=177 parameters. No state index or per-state gate is allowed.

The existing softplus energy and C C^T tensor make E>=0 and s>=0 mathematically; thus both log1p domains are valid. Finite checks must reject nonfinite features/outputs or a violated domain, preserving failure evidence rather than removing rows or changing epsilon. Floating-point underflow to zero is allowed. No labels or mask values enter the gate, B, q or b. Missing-label synthetic cases may test loss masking, but must not change which predicted slots form the gate.

E'_k=E_k in the forward pass. Native f'_k=2*E'_k*tr(A'_k)/(3*27.211386245988), with no clipping, cap, teacher energy or calibration. Keep E, A, B, q and all features differentiable. After the gate's initially zero final layer moves, trace loss can reach the energy head through mean(E_H) and sum(E_H*s), in **both** modified arms. Energy changes therefore cannot be attributed solely to directional tensor context. No detach or frozen energy branch is introduced.

Tensor strength expands as s'_k=s_k+2b tr(B A_k)+b^2 tr(B^2 A_k). This gives state-dependent invariant tensor-overlap information. The scalar arm uses the same molecular features, including anisotropy q, but only a common strength multiplier. Both modified arms satisfy ||bB||_F=||bqI||_F. Equal parameter count, features and perturbation norm do not match every output gain, effect range or function class; these limitations are part of interpreting the comparison.

## 3. Initialization, identity and symmetry contract

Rebuild the fresh seed11 base without opening any historical checkpoint. Its expected base tensor hash is `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; the inherited disabled-F state before adding the new gate has hash `3c5d20463a6aa1d4e7c25ee5e57e0ceaa53c17e5a89af35d7761b58e58d661f9`. Construct the gate in a separate forked CPU RNG context with seed11, standard Linear initialization for its first layer, and set the final weight and bias to exact zero. The three arms share the entire initial base and gate tensors. The measured common full-schema hash is `e601a03737fc09da857f71238f193a6511d58360b05404c3a8cae1d5b84dd9d7`; gate hash is `8a4c3581043b25ac533658a7ee9f6402fa35d9a3bc59a60b9f6409cb518d4d3e`. Constructor checks establish CPU RNG isolation. `torch.manual_seed` can also reset initialized CUDA generators; production calls common setup after construction and restores saved RNG after resume. The independent data-order generator is unaffected, and the fixture verifies exact saved/restored CPU/CUDA RNG and order. Constructor isolation alone is not a claim about every CUDA generator. All disposable fixture state is excluded from production initialization.

At initialization b=0 and every arm returns the original A/E/f. No `if b==0` training shortcut is allowed, because that would suppress the gate derivative. At fixed incoming A,

    d tr(A'_k)/db at b=0 = 2 tr(A_k B)   (tensor)
    d tr(A'_k)/db at b=0 = 2 q s_k       (scalar)

These derivatives are positive for nonzero PSD A. A controlled nonzero-error synthetic fixture must demonstrate live final-layer gradients in both arms. Earlier gate-layer gradients are initially zero by the intended zero final weights; they should become live after a hand-set nonzero final layer or an authorized update. Zero target/error cancellation is not a universal architecture failure.

In exact arithmetic B is positive definite with trace1; q is between 1/3 and 1/sqrt(3). Since |b|<=.25, tensor L=I+bB is positive definite with eigenvalues bounded by .75 and1.25. Scalar L has the tighter corresponding q bounds. Congruence preserves PSD and rank at fixed incoming A; it cannot revive exactly zero A. The base remains trainable and can change its incoming rank/strength. Both modified arms coincide for isotropic aggregate R, including R=0. All strengths move in the sign of b at fixed inputs; this is not arbitrary transfer of brightness among states or a sum rule.

For any proper or improper O, A→OAO^T, B→OBO^T and L→OLO^T, while x and b are invariant. Therefore A' is O(3)-covariant, f invariant, including reflections and inversion. The module inherits translation and atom-permutation invariance from the geometry backbone and molecular counts. It does not force a planar normal tensor component to vanish. It depends only on A, so mu→-mu and independent right-orthogonal factor changes C_k→C_k V_k leave it unchanged.

**Exact degeneracy claim is limited:** in a real dipole outer-product fixture with orthogonal state mixing inside a strictly equal-energy block, the block sum of A, aggregate R/S and energy-weighted sum remain invariant. The common L then gives the corresponding transformed block-sum invariance and unchanged outside-state outputs. Individual state strengths may change. This is not a guarantee for arbitrary transformations of learned trace-supervised factors, complex electronic characters, near-degenerate ordered roots or the benchmark's per-state labels. Never swap, merge, match or exclude labels in primary evaluation. No TDDFT X/Y, density, NTO or AO-integral interpretation is claimed.

## 4. Fixed objective, split and label boundary

All three arms use coefficient1 for each unchanged original term:

    LE = mean((E_pred[mask_E] - E_target[mask_E])^2) / sE2
    Ls = mean((tr(A'_pred)[mask_E & mask_A]
               - tr(A_target)[mask_E & mask_A])^2) / (3*sA2)
    objective = LE + Ls

Original linear trace extraction may precede valid indexing as in the reviewed base_loss; valid selection precedes nonlinear error/square arithmetic. Empty valid sets fail. The frozen corpus has all ten E/A/f masks true on every assigned TRAIN/validation molecule; assert this contract and fail on disagreement rather than silently excluding labels. Printed-f targets, including all valid zeros, are used for evaluation and no-gradient diagnostics, not a new training objective. Gate features always include every predicted slot. The return E/A used for loss and evaluation is the transformed output; optional untransformed diagnostics never enter the objective.

Use the exact v2 TRAIN statistics `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`: sE2=.5376833706691944, sA2=.13245475393297818, original per-state energy means and n_ref18. No new fitted normalization. Split manifest `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`, independent verification `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`: TRAIN120355 / validation6686 / TEST6686.

Reuse the reviewed selected-row decoder and audit actual fields/index hashes before numerical conversion. Full-corpus target decoding followed by slicing is forbidden. TEST numeric targets remain sealed; raw-byte hashing/ZIP transport is distinguished from decoding. No new split or label/geometry exclusion is allowed. All66,860 validation labels remain in pooled and per-state scoring. The v2 TEST contains5989 old-TRAIN/361 old-validation/336 old-TEST rows; the partitions are disjoint under the audited conservative identity rules, not an absolute chemical-identity proof. This historically exposed repartition is not external fresh confirmation. Fresh base initialization is mandatory.

## 5. Matched budget, controls and diagnostics

Common settings: seed11, independent PCG64 order_seed11 and all60 inherited exact TRAIN order hashes; batch64 including the final short batch; original Adam AMSGrad, fixed LR.001, betas(.9,.999), eps1e-8, WD0, global clip5, FP32/noAMP/noTF32, two CPU threads. No scheduler, early stop, gradient matching or arm-specific change. All base parameters plus the enabled gate are trained from scratch for exactly60 epochs,1881 batches/epoch,112860 updates/arm. No trained Round05/06 state or technical fixture checkpoint is reused. Identical starting tensors/order do not imply bitwise CUDA trajectories.

Use the same runner, loss and diagnostic code path for the three arms, with the explicit transform mode as the only branch. No gratuitous zero-weight autograd term in `original`. Log no-gradient diagnostics consistently: gate b/q/input-range summaries, relative ||A'-A||/max(||A||,1e-12) and strength ratio with zero denominators reported separately, gate/base gradient norms, clipping, LE/Ls, raw-f MSE and timing/memory. The denominator floor is diagnostic only; no normalization in the predictor changes. TRAIN values are pre-update minibatch trajectory aggregates, not fixed final-checkpoint TRAIN evaluations. Do not perform extra TRAIN inference or compare such aggregates as a matched generalization gap.

A prior control rerun moved from .447169 to .404798 with matched scientific recipe/initialization/order but different known diagnostic workloads/concurrency. It has no isolated cause. Therefore all three controls must be contemporaneous; the old reference is an additional safeguard. New diagnostics must be identical across arms and logged without changing the objective. Model metadata must distinguish 177 active modified-arm parameters from the frozen original-arm schema.

Proposed allocation: physical GPUs1/2/4 for original/scalar/tensor, sequential fresh admission with separate UUIDs/shared locks before imports, then concurrent runs. No fallback, GPU0 interference or use of excluded3/7. A later launch decision must bind exact healthy-device policy/registration and preserve the existing four-hour monitor. Round06 took approximately2.60–2.62h/arm; budget roughly8 GPU-hours, allow25% operational headroom (~10GPU-hours/~3.3h wall time on three available GPUs). Measure actual overhead later; these are estimates, not an instruction to extend60 epochs. The transform is O(10*3^3) work per molecule plus177 gate parameters.

## 6. Validation selection, reporting and allocation rule

Evaluate complete validation at epoch0 and every completed epoch. Select each arm by earliest minimum pooled native raw-f SSE, equivalently maximum R² on fixed labels. Compute f in FP64 from predicted E/A' with `2/(3*27.211386245988)` and authoritative printed targets; no calibration/average/clamp. Training remains FP32. Report selected-best and aligned epoch60 separately, all ten state errors, energy error, TRAIN q90/q99=.0549/.2406 true-tail counts/RMSE/MAE/SSE, disjoint true/predicted q99 bins, prediction/error quantiles and top-error SSE share. No diagnostic changes the checkpoint selector.

**Tensor-specific allocation gate:** selected tensor R² must be at least .003 above EACH of (a) selected contemporaneous original, (b) selected contemporaneous scalar, and (c) retained v2 reference .44716940136585204. The reference component therefore requires at least .45016940136585204. Passing grants eligibility only for a separate root decision on paired fresh independent seeds23/37 with these same three arms; never average their predictions. Report individual seed outcomes. It is not statistical significance or fresh holdout confirmation.

If the tensor gate fails, close the fixed three-arm study and publish it; no same-settings extension, epsilon/bound/LR sweep or extra seed is automatic. Report scalar-minus-original and every arm-minus-retained. A scalar improvement can become a descriptive single-checkpoint reference, but it is not evidence for directional tensor context and does not pass the tensor gate. Any later scalar-specific allocation requires an explicit distinct root decision; this proposal grants none. A stronger original rerun similarly updates the descriptive reference only. Do not ignore a better control, and do not average repeat controls.

If intervals are reported, predeclare paired identity-component bootstrap seed20260930/2000 draws, resample entire validation components/all states and recompute pooled SST. Include tensor-vs-original, tensor-vs-scalar and scalar-vs-original; these are descriptive, conditional on selected checkpoints/reused validation. No interval versus the aggregate-only retained reference without its saved paired inputs. No old/new TEST inference or labels for selection. Any final TEST decision remains separately authorized after a locked recipe and independent-seed review.

## 7. Completed bounded preflight contract

The root preparation decision and independent exact-source reviews preceded execution. CPU prebinding was empty CUDA, OMP/MKL2, before imports. Synthetic tensors/targets/geometry covered original loss/function parity; zero/identity and isotropic equality; proper/improper O(3), atom permutation/translation; independent C-factor gauges; real equal-energy dipole-block rotation; PSD/rank/bounds and planar/inversion fixtures; scalar/tensor perturbation norm; finite zero/tiny/bright-strength features; exact parameter/init hashes; live final-gate and base gradients plus later first-layer gradients. Hand-set tensors tested the 1/(S+epsilon) derivative near zero without selecting a new epsilon. Invalid-target sentinels tested loss masks while gate inputs remained all predictions. No historical/trained checkpoint or dataset was opened by this CPU fixture; its own temporary synthetic geometry export was the sole allowed model file.

Fixed numeric criteria: module algebra/analytical derivative comparisons in FP64 atol1e-10/rtol1e-8; FP32 algebra/gradient comparisons atol2e-6/rtol1e-5. Exact identity on the same precomputed incoming E/A and exact base/gate initialization/RNG/order/integer states. Live derivative fixtures require finite positive expected derivatives and nonzero final-gate gradients; no universal claim for arbitrary zero residuals. Failures must be recorded and reviewed before any authorized retry; no outcome-dependent tolerance relaxation.

The bounded GPU check ran once after exact source review and fresh GPU1 admission/owned registration: first128 ascending new TRAIN rows, consecutive64+64 batches; each arm executed update1, update2, restored update1 and replayed update2. **Exactly9 discarded optimizer updates total**, no extra rows or fit. It used the actual production objective and retained graphs for gradient diagnostics without additional updates. Full forward/export used only the first two already decoded TRAIN geometries. Resumed model/Adam criteria were atol2e-6/rtol1e-5, loss atol1e-6/rtol1e-5; RNG/order exact. Full CUDA original-versus-wrapper E/A parity was checked only at identity initialization. After updates, each transformed model was compared with its own geometry export; its nonzero-gate outputs intentionally need not equal the original model. Both parity checks used the fixed atol2e-6/rtol1e-5, reporting native-f differences. The observed small CUDA differences do not establish whole-trajectory bitwise determinism. Tiny early rows give indicative memory/timing only.

The synthetic hand-set gate passed one-checkpoint CPU load-time and forward access auditing before the TRAIN fixture; geometry export contains original base+one transform+TRAIN stats/config/buffers and the exact transform contract, with no old checkpoint, label/cache/dataset file access. The GPU fixture checked its trained disposable export; it may never seed production. Twenty-two reviewer-owned synthetic authorization checks include rejection before scientific imports/data/artifacts. Completed-stage refusal and exact-source recovery are mandatory. CPU/GPU receipts and independent reviews bind the exact executed bytes; neither completed numerical stage may be repeated as routine verification.

## 8. History, physical limits and next decision

Round01 adapted selected eta0 and all arms selected epoch0; Round02 froze it and learned F with tiny failed gains. Round05 jointly trained fresh original/right-F/raw-M-decor/both, but F/decor/both failed the original-control improvement gate. Round06 changed only the intensity objective on fresh original PSD: .422028 vs contemporaneous .404798, below retained .447169; it failed the dual gate and showed mixed tail/energy tradeoffs. Scalar calibration/f-only spline routes also failed to improve their eligible references. These are negative evidence against assuming that another small readout will reach .60; they do not test this A-dependent common spatial transform.

Historical .373493 was a direct-f head with a raw-f objective and altered head/initialization, not the original PSD LE+Ls baseline. Do not recycle that mislabeled comparison. This proposal returns to the common original objective and adds only the prespecified transform, isolating the next question as far as its controls permit. Existing A already encodes coherent interactions; no missing transition-density mechanism is asserted.

Under trace supervision, orientation remains unidentifiable from labels. Correlated learned A directions, phase-blind aggregation and a bounded common response are empirical inductive biases; they are not recovered electronic amplitudes, AO density, NTOs, oscillator sum rules or full-TDDFT X/Y. The scalar arm already sees q; a tensor advantage would concern invariant overlap information under this parameterization, not unique QC physics. The bound may leave large false-bright errors unresolved; nonfinite conditioning and repeated-run variability remain risks. No ordinary anonymous polar-vector or naive AO transition-density substitute is authorized.

Implementation and the nine-update technical scope are complete; final preparation review and publication remain separate gates. FIRST archive lightweight records under D:\MTO\archives, then inspect/commit/push; require a separate bound root execution authorization for three60-epoch fits. Keep all learned gate/model/optimizer tensors, datasets, split/identity/prediction arrays and caches server-only. There is no open numerical setting to tune from data. The pending decision is production allocation after preparation evidence, not an invitation to search. No empirical validation gain or physical interpretation is established by the technical checks.

## 9. Future production commands and recovery boundary

All production commands below are currently blocked. `PRODUCTION_AUTHORIZATION_TEMPLATE.json` has `authorized:false`; it is not an execution decision. A later root-issued `PRODUCTION_EXECUTION_AUTHORIZATION.json` must set `authorized:true`, scope `round07_three_arm_60epoch_fit`, ordered arms `["original","scalar","tensor"]`, epochs60, TEST access false and historical weights false. It must bind the exact frozen manifest, independent preparation review, v2 split/verifier and publication receipt bytes. The receipt path is `/home/inspur/MTO-1/research/single_model_20260929/ops/ROUND07_PREPARATION_PUBLICATION_RECEIPT.json`; its verified-remote/archive-first flags and manifest/review hashes must match. No preparation acceptance or publication alone grants execution.

After that separate authority exists, the sole executor uses:

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /usr/bin/python3 launch.py \
  --authorization /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json \
  --arms original scalar tensor
```

The stdlib launcher verifies authority before run artifacts, admits physicalGPU1/2/4 sequentially under separate shared locks, then starts original/scalar/tensor concurrently. Each child receives its UUID before the pinned Python interpreter imports CUDA modules, plus two threads and unbuffered logs. UUIDs are `GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800`, `GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa`, and `GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184`. Logical CUDA0 maps to the assigned physical device. Registry receipts pin PID/start/boot/UID/cwd/argv, exact source/authority and GPU identity. Existing four-hour monitoring continues unchanged.

If an admission or child fails, preserve its attempt/log/terminal evidence and inspect the exact state; do not blindly rerun the command or move to another GPU. A live worker lock and a completed marker both refuse relaunch. After an explicit recovery decision, `--arms <one incomplete arm>` uses the same bindings and its committed `last.pt`; no completed arm may run again. `last.pt` commits optimizer/RNG/order/history at each completed epoch, so a mid-epoch interruption replays that epoch from its previous commit. `best.pt` is the validation-selected model/optimizer snapshot, while `last.pt` is this runner's resume entry. `geometry_best.pt` is the selected standalone inference artifact. Technical fixture parameters and optimizer states are never recovery inputs.

Geometry-only deployment loads one `geometry_best.pt` with `predictor.load_predictor(path, device)` and calls `model(z,pos,batch,n,edge_index)`, where `n` is the molecule count and `edge_index` is optional. Its embedded full base, one gate, stats/config, readout mode, transform-contract metadata and buffer fingerprint are strictly verified without reopening source checkpoints, raw data or caches. No inference command is authorized during this preparation closeout.
