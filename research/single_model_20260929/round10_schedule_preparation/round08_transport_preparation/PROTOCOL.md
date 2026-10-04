# Round08 implemented preparation contract

Root decision a227764a9540cfa4f89dd9ed8ea48aa923d65bb4152f2e90ff734073601379a1 authorizes implementation and independently reviewed CPU/GPU technical checks, including exactly nine discarded first128 TRAIN updates. CPU synthetic checks and the prescribed nine-update GPU stage have passed. Final independent closure review remains pending. Full60-epoch fitting requires that review, root acceptance, D-first reviewed publication and distinct bound execution authorization. PROTOCOL_PROPOSAL.md preserves the accepted proposal verbatim. TEST remains sealed.

Implementation adds transport only inside core blocks2/3; it keeps original parameter keys and common fresh base initialization. Existing Attention is evaluated once per block and its actual mijs2 is reused. Original and active wrappers share the intended update ordering; active residual uses pre-block T. The original arm bypasses the residual with frozen zero theta. Scalar coefficients are shared over magnetic components, exactly768 trainable added parameters per active arm.

Diagnostics explicitly report regularized relative message/pre-block norms using denominator max(norm,1e-12), with zero and below-floor counts. They never rescale activations or gradients. Atom sums/counts produce atom-weighted epoch norms; raw-M/loss/gradient summaries are weighted pre-update minibatch trajectory aggregates, not fixed-checkpoint TRAIN evaluations. Loss remains original LE+Ls. Native E stays softplus positive and A PSD; no output transform or raw-f objective is introduced.

Geometric inputs are z,pos,batch,n,edge_index only. Strict one-checkpoint export binds transport mode/contract and full base state/config/TRAIN statistics. Inherited resume semantics keep atomic last.pt as the committed epoch boundary; best.pt is the selected training snapshot. No preflight state initializes production. CPU constructor RNG isolation does not claim universal CUDA-generator preservation; production setup runs after construction and explicit RNG restore follows resume.

The exact formulas/settings/selection below are unchanged. No production authority comes from this document.

---
# Explicit neighbor tensor transport inside the geometry backbone

This is the accepted contrast arising from `INFORMATION_FLOW_AUDIT.md`, after completed Round07 publication703c2cd7. Its implementation and fixed technical checks fall within the bounded preparation decision; production remains separately gated. It is not a route guaranteed to reach.60.

## One falsifiable question and controls

Does carrying an atom's existing oriented tensor features across an edge improve pooled raw-f generalization beyond an equal-parameter local tensor residual and an unchanged original backbone, with the same MTO/PSD decoder and original LE+Ls?

| Arm | Added backbone update | Added trainable parameters |
|---|---|---:|
| original | Zero; dormant schema only |0|
| local | Receiver's own T, gated/averaged over its incoming edges |768|
| neighbor | Source T from each incoming edge, same gates/averaging |768|

The local control matches parameter count, insertion sites, gate inputs, degree normalization and identity initialization. It does not match the function class, activation distribution or effect range of neighbor transport. A neighbor win tests the complete spatial-transport recipe; it cannot isolate a physical mechanism or eliminate optimization effects.

## Exact operation

Keep original DetaNet128 channels, three blocks, irreps1o/2e/3o, radius/edge list, radial functions and all existing weights/updates. Modify only blocks2 and3. Block1 begins with T=0, so its proposed transport would have zero learning signal and is omitted deliberately.

Use the code's actual receiving convention: `i=edge_index[0]`, `j=edge_index[1]`; the existing Update scatters messages into j. For each block b∈{2,3}, irrep l∈{1,2,3}, channel c∈{1,…,128}, magnetic component m, take **pre-block** T and the existing scalar edge coefficient a_ij,c=`mijs2` before its original scalar×harmonic tensor product. Define d_j as the number of incoming preserved edges, and:

    G_ij,c = tanh(a_ij,c)
    H_neighbor[j,l,c,m] = sum_(i→j) G_ij,c T[i,l,c,m] / max(1,d_j)
    H_local[j,l,c,m]    = sum_(i→j) G_ij,c T[j,l,c,m] / max(1,d_j)
    deltaT[j,l,c,m] = tanh(theta[b,l,c]) H_mode[j,l,c,m]

An empty incoming neighborhood contributes zero. No self-edge is inserted or removed. There are2×3×128=768 scalar theta parameters. Initialize all theta=0. The original arm stores the same zero theta tensors frozen and bypasses the added path; local/neighbor execute the differentiable path without a theta==0 shortcut. Both use the same zero initialization. The original edge coefficient/network is shared with its harmonic message; no extra MLP or equivariant linear matrix is introduced.

Insert deltaT **after** `outt(scatter(mijt))` and before the existing local tensor-product attention:

    T_mid = T_old + original_ut + deltaT
    S_mid = unchanged original scalar residual
    (ut2,us2) = original local update attention(T_mid,S_mid)
    T_new = T_mid + ut2; S_new = unchanged remaining scalar update

No detachment of T or edge gates; all original parameters remain trainable. No normalization over magnetic components, tensor rotations, learned frame, amplitude clipping, new state features or loss term. Degree averaging keeps the new sum from scaling linearly with neighbor count; bounded scalar coefficients do not guarantee bounded network activations or gradients.

At theta=0 the transform is exactly nested at the message/update algebra level and d(deltaT)/dtheta=H. Nonzero pre-block T and nonzero edge gate can give live theta gradients; symmetry or cancellations can make particular channels zero. Both modified arms have the same potential issue. Preflight measures live gradients on the fixed fixture, without tuning an amplitude to achieve a desired norm. Tensor/edge-gate base gradients through the new branch begin zero and become live after theta moves; the original paths are already live.

## Mathematical and physical boundaries

T is stored in a common global O(3) irrep convention. Multiplication by invariant scalars, same-irrep addition and permutation-consistent edge sums commute with rotations and inversion; relative geometry preserves translation invariance. Channel-specific theta is shared over every m. Therefore the added path is an equivariant interatomic feature transport, without treating T as a transition dipole or density.

Neighbor orientation already affects scalar messages indirectly through original T→S contractions. The new path bypasses that invariant compression; it does not prove an otherwise unrepresentable target, extend graph cutoff, add a new interaction block or guarantee a useful long-range response. No transition phase, orbital orthogonality, oscillator sum rule or degenerate-state covariance is imposed. Output PSD/softplus guarantees and existing physical-interpretation limits remain unchanged.

## Fixed matched budget and selection

- Same audited v2 split and TRAIN statistics as Round05–07; no newly decoded TEST target. TRAIN120355/VAL6686 with all ten valid slots and zeros retained. Current q90=.0549/q99=.2406 remain fixed diagnostics. v2 is disjoint under audited conservative identity rules but historically exposed, not external fresh data.
- Fresh original-base seed11, order11 and exact common original tensor initialization hash; no historical trained checkpoint or discarded preflight state. Add zero transport tensors after base construction without shifting base/data RNG. Dormant right-F remains disabled/frozen; no congruence output correction, decorrelation penalty, direct-f head or calibration.
- Original LE+Ls, same valid masks/reductions/TRAIN normalization, predicted softplus E and PSD A. Native f=2E_eV tr(A)/(3×27.211386245988). All parameters except dormant schema active as declared.
- Fixed60 epochs, batch64,1881 updates/epoch/112860 total; Adam AMSGrad LR.001,betas(.9,.999),eps1e-8,WD0,clip5,FP32,noAMP/noTF32,2 CPU threads. No schedule, early stop, adaptive scaling or fallback. All arms use one common implementation for loss/evaluation/diagnostics, while their intended transport branches differ.
- Earliest minimum pooled native VAL raw-f SSE among epochs0–60 chooses each checkpoint. Report selected and aligned60 outcomes, all states/energy/true tails/four false-bright bins and full counts. No historical or v2 TEST inference, model averaging, label permutation or selective exclusion.
- Preserve atomic last.pt with optimizer/RNG/order/committed epoch and source/split bindings; best.pt selected snapshot; one self-contained geometry export with embedded transport contract/mode/config/TRAIN statistics and strict load-time/forward access checks. Complete-stage refusal and explicit reviewed recovery only.

This is declared-setting/base/order matching, not a promise of identical CUDA trajectories. Preserve the prior repeated-control caveat and use identical diagnostics across these three arms as far as the intended mode allows. A source/gate failure is engineering evidence, not an automatic license to restart or change the recipe.

## Fixed allocation and interpretation rules

For **neighbor-specific seed allocation**, require selected neighbor R² ≥ selected original+.003, ≥ selected local+.003, and ≥ retained.44716940136585204+.003. Treat this as a screen for a later independent-seed study, not a robust architecture claim. No score-based extension or alternate checkpoint rule if it fails. If local or original beats the retained reference, preserve the better control recipe/checkpoint descriptively; that does not establish tensor transport and grants no automatic new allocation.

If the screen passes and root separately authorizes confirmation, repeat the complete three-arm contrast from fresh seeds23 and37 with their own paired data-order contracts; report each checkpoint separately and require the candidate advantage over both paired controls in each seed before a broader claim. Never average predictions. Existing component-bootstrap intervals may describe conditional selected-VAL uncertainty but do not substitute for training-seed confirmation. No TEST release follows automatically.

Record per-state/tail/energy tradeoffs at the same selected checkpoint; substantial regressions require an explicit next decision rather than post hoc coefficient or checkpoint tuning. The sole automatic screen is the pooled triple gate above. Thresholds do not establish absence of harms or correct multiple-selection/reused-validation uncertainty.

## Authorized bounded preparation and remaining production gates

1. Pure synthetic CPU checks of standalone branch O(3)/reflection, translation/permutation, exact theta0 nested algebra, edge receiver convention, degree/empty-neighborhood behavior, and analytical theta gradients. Demonstrate nonlocal dependence by changing a synthetic source T at fixed receiver/edge gates: neighbor update changes while local does not. This tests the implemented path, not real data expressivity.
2. Common original initialization/base/order hashes, no inherited trained weights; exact parameter count and original-mode zero state. End-to-end identity versus original and each nonzero transformed model versus its own one-file export. Test load-time and forward-time external access; model inputs only z/pos/batch/n/edge_index.
3. Fixed bounded engineering fixture: same first128 newTRAIN rows/two64 batches, three executed updates per arm (uninterrupted first+second and replayed second), **nine discarded updates total**. No VAL/TEST numeric decoding; stream selected TRAIN rows only and instrument actual fields/indices. The numerical bounds were fixed in TECHNICAL_PREFLIGHT_PLAN.md before execution, preserving exact RNG/order checks; no outcome-driven tolerance changes. Any fixture state is private/discarded and never enters production initialization.
4. During eventual production log theta gradients/movement, transported-message versus original-message norm by block/irrep, and usual losses/clip fraction. Aggregates along TRAIN updates do not count as a final-checkpoint TRAIN score or evidence of physical transport. Do not add extra backward probes or tune loss/scale from these diagnostics.
5. Independent source/result review, source freeze and D-first reviewed publication, then separate bound root execution authority and immediate healthy-GPU lock/UUID/registration. Publication or this preparation alone cannot launch a fit. The inherited registration race is corrected by a standard-library pipe barrier: the owned worker cannot import the scientific child until registration succeeds. Failure closes the barrier and stops only that owned worker; no automatic retry.

## Cost and choice

Existing original60-epoch fits cost about2.6–2.83 GPU-hours each. This three-arm pilot has an8–9 GPU-hour baseline cost before the extra gather/scatter of128×15 tensor components in two blocks. Provision conservatively for **up to12 GPU-hours**, about3–4 wall-hours with three available healthy GPUs. This is a planning estimate, not measured performance; actual memory/runtime admission must come from the bounded authorized fixture, whose small molecules do not guarantee worst-case batch memory. No resource-driven change to scientific budget is automatic.

Two later three-arm confirmation seeds would add roughly16–24 GPU-hours and require a new decision. The proposed contrast costs more than a weight-decay toggle, but directly tests a source-identified communication path with a capacity-matched local control. It is smaller and more interpretable than adding depth/width/state-conditioned blocks or an unvalidated AO/QC representation simultaneously. It may still fail; negative outcomes should close this exact transport choice without searching coefficient scales or neighborhoods.

Root has selected this question and its fixed settings, nine-update preparation scope and triple+.003 screen. Preparation acceptance, publication and bound production execution authority remain separate decisions. Passing synthetic or technical checks demonstrates implementation behavior on the declared fixtures; it establishes neither empirical accuracy gain nor a physical mechanism or worst-case training-memory guarantee.
