# QC follow-up: a common transform of phase-blind state tensors

**Recommendation:** after Round05 evidence, consider one small empirical readout study: a common, geometry-dependent congruence of the existing PSD state tensors, compared with an isotropic transform using the same nonredundant gate. The synthetic algebra passes. This note authorizes no implementation in production, fit, validation selection, or test access. Round05 preparation remains the priority.

This replaces the unfinished two-output discussion in the draft. Two scalar outputs entering only one isotropic scale would not be an active capacity-matched control. The final design below uses one output in both arms.

## What changes, and why it is a distinct hypothesis

The actual original `PhysicalSpectrumDecoder` already forms a coherent channel sum Q, then C=beta I/sqrt(3)+Q and A=C Cᵀ. Its PSD output range and scalar strength are not missing expressivity. Its per-state readout lacks an explicit dependence on the other states' predicted spatial tensors.

The new bias is that **all ten states share one bounded Cartesian response transform inferred from their aggregate tensor**. A transition aligned with a strong aggregate direction receives a different correction from another direction. This couples states through their tensor orientations, without assigning signs to transition vectors or mixing arbitrary C square roots. It can change molecular total strength, unlike a pure orthogonal state mixer. The construction is an empirical hypothesis, not an identified Hamiltonian, TDDFT response kernel, transition density, NTO basis, or sum rule.

Most importantly, eta0's LE+Ls supervises E and tr(A), not measured tensor directions. A's orientation may be a useful **latent feature** but is not identified physical polarization. The new loss remains LE+Ls; no directional labels are added. Directional supervision would require a separate frame audit and matched objective controls. The existing data/frame audit supports Input-frame provenance but has not directly verified vector alignment.

## Exact proposed readout and control

For predicted state tensors A_k and predicted energies E_k, k=1..10, use atomic units for strength and convert energy to Hartree with the existing constant 27.211386245988 eV/Hartree:

    s_k = tr(A_k)
    R = sum_k A_k;  S = tr(R)
    epsilon = 1e-6 (atomic-unit squared electric dipole)
    B = (R + epsilon I/3)/(S + epsilon)
    q = sqrt(tr(B B)/3)
    b = 0.25 tanh(g(x))

The same nine invariant gate inputs in both arms are:

1. Five values log(1+N_Z), Z=H,C,N,O,F.
2. mean_k(E_k/Hartree).
3. log(1+S/s0), with fixed s0=1 atomic-unit squared dipole.
4. log(1+sum_k[(E_k/Hartree) s_k/s0]).
5. q.

These are predictions and geometry-derived counts, never QC targets at inference. The fixed unit scales are not fitted to old data. `g` is Linear(9,16), SiLU, Linear(16,1): **177 parameters**. Initialize its last weight and bias to zero and share the exact initial gate tensors across arms. Use the same fresh base tensors, optimization, objective, order and selection policy. Keep all ten output slots; masks apply to supervised losses, not to selecting gate inputs.

**Gate domain verified from actual source:** `frozen_reference/models_ea.py:MTOEA.forward` and current `architecture/model.py:AdapterMTOEA.forward` both use `E=F.softplus(energy_head+energy_offset)` and return A=C Cᵀ. For finite decoder outputs, E is nonnegative (strictly positive mathematically; floating-point underflow may give zero) and s is nonnegative. Therefore sum(E_H s/s0)>=0, so the specified log1p has a valid domain without any clamp or empirical range assumption. Nonfinite decoder outputs must fail preflight/run finite checks, not be silently repaired. Keep E/A/gate features differentiable without stop-gradient. Source hashes and the primary-paper verification are recorded in `qc_response_congruence/SOURCE_RECEIPT.json`.

| Arm | Common matrix L | State output |
|---|---|---|
| Tensor candidate | I+bB | A'_k=L A_k Lᵀ |
| Isotropic control | (1+bq)I | A'_k=L A_k Lᵀ |
| Original reference | I | A'_k=A_k |

Both modified arms retain E and compute native f'=2 E_Hartree tr(A')/3. Broadening and label order stay unchanged. One model/checkpoint contains the whole predictor. No calibration composition or averaging.

The two modified arms have identical gate inputs, active gate parameterization, identity initialization, and perturbation norm: ||bB||_F=||bqI||_F. They coincide exactly for isotropic R. The control can use aggregate anisotropy q but cannot use its orientation relative to each A_k. This tests directional context beyond a molecular scalar rescale. It does **not** make the two mechanisms equally expressive or match every eigenvalue/gain range; those differences are the tested restriction.

## Mathematical behavior

- **PSD and boundedness:** B is SPD, tr(B)=1, with eigenvalues in (0,1). Thus L is SPD/invertible with eigenvalues between 0.75 and 1.25; q lies in [1/3,1/sqrt(3)], giving tighter control bounds. Congruence preserves PSD and rank. It cannot turn a full-rank A into a pure rank-one transition or revive an exactly zero A by itself. All base parameters remain trainable in any later scratch pilot.
- **Near zero:** epsilon has the same units as S and makes R=0 give B=I/3, q=1/3. Tiny tensors smoothly approach isotropy instead of selecting a direction from numerical noise. Epsilon is a fixed numerical floor, not a fitted physical constant. No matrix inverse, eigenvector, learned eigensystem, inverse energy gap, or normalized single dipole appears. Although dB/dR scales with 1/(S+epsilon), the complete output multiplies A_k; actual FP32/autograd conditioning still needs preflight.
- **O(3), atoms and origin:** under any proper or improper O, A→OAOᵀ, B→OBOᵀ, L→OLOᵀ. Counts/E/q are invariant, so output covariance follows. An atom permutation leaves molecular inputs unchanged. The readout has no position/origin input; it inherits translation invariance from the backbone. A planar mirror permits nonzero A_zz, as does inversion; neither forces a bright transition to zero.
- **Phase and factor gauges:** A is unchanged by mu→−mu and by any independent right-orthogonal factor change C_k→C_k V_k. This readout depends only on A and is invariant to all such factor gauges. It does not resolve the separate lack of physical orientation supervision.
- **Exact degeneracy:** for an explicit dipole-amplitude fixture, an orthogonal rotation within a block of equal energies preserves that block's sum of A, R, S, energy-weighted strength sum and all gate inputs. The same L therefore preserves covariance of the transformed block sum and leaves outside states unchanged. Individual tensors/strengths generally change and are not invariant targets. This is a block-sum property, not recovery of electronic state character or the benchmark's arbitrary degenerate-root assignment. Near-degenerate ordered labels remain potentially difficult; smooth readout arithmetic does not solve crossings. Never merge, swap, or exclude labels in primary raw-f scoring.
- **Live initialization:** at b=0, d tr(A'_k)/db=2 tr(A_k B) for the candidate and 2 q s_k for the control. Both are positive for nonzero PSD A. The final gate layer has a live derivative on generic nonzero-error fixtures; earlier gate layers initially have zero gradients because the final weight is zero. Zero targets or cancellation can yield zero loss gradients, but neither arm has the universal U-squared-at-identity dead derivative. Increasing/decreasing b changes all strengths in the same sign, with directional differences in magnitude; it is not unconstrained state redistribution.

## Synthetic evidence and remaining checks

`qc_response_congruence/synthetic_preflight.py` ran locally with NumPy only, seed20260930. No real labels, model outputs, checkpoint, GPU or fit was used. `SYNTHETIC_RESULTS.json` records:

| Fixture | Result |
|---|---:|
| Proper/improper rotations, inversion, state permutation | max error <7e-16 |
| Independent factor gauges | 1.8e-15 |
| Identity and isotropic candidate/control equality | exact |
| Matched perturbation Frobenius norm | 1.2e-16 |
| Exact-degenerate block output sum | 4.5e-16; individual outputs correctly change |
| Neutral synthetic charge origin shift | 9.2e-15 |
| Finite-difference gate derivative | max error <1e-9; both arms live |
| Planar normal transition | remains nonzero; mirror-invariant |
| Zero/isotropic/tiny strength | finite |

The origin fixture verifies neutral synthetic dipoles, not a proposed point-charge model. Eigenvalues were used only to inspect synthetic PSD outputs, never to construct or train the readout. The full gate, real MTO integration, FP32/autograd and resume behavior remain **unverified**. This is an algebra PASS, not a production preflight PASS.

## Budget and decision gates

1. **Now completed:** sub-second local algebra check and read-only review. No data-dependent inference.
2. **Only after a separate preparation decision:** port the two exact formulas and 177-parameter gate into an isolated namespace. Budget at most one CPU hour for synthetic Torch gradient/gradcheck, bounds, gate-input transformations and export checks. After the v2 split is frozen, separately authorize one fixed TRAIN-only two-update/resume fixture, at most 30 GPU-minutes. Check both gate gradients and all-base backprop, masks, element counts, no old statistics/weights, exact base/gate initialization, one-checkpoint parity, and no test decoding. Resolve failures scientifically before any production authorization. Added readout work is O(K·3³) and O(K·3²) storage, not O(K³) state diagonalization or AO-pair covariance storage; measure actual overhead rather than assume it.
3. **Conditional later pilot, after Round05 review/publication:** fresh original reference + isotropic control + tensor candidate, common seed/order11, original LE+Ls and the same fixed60-epoch recipe on the frozen v2 split. Proposed budget ~7.5–9 GPU-hours before measured overhead/headroom. Do not initialize from the old incumbent or select a modified base from Round05 silently. Require candidate selected-validation ΔR²>=0.003 over **both** contemporaneous original and isotropic control for any further allocation; inspect aligned epoch60, state/tail/false-bright and E errors. No selective extensions after a failure; no claim that a 60-epoch null proves all such operators useless. A promising result needs a separately reviewed paired-seed confirmation, each predictor reported separately, before any frozen final test decision.

There is **no fit gate passed now**. Round05 may change whether this follow-up is worth allocating at all. The new partition remains historically exposed data, not fresh external evidence. The target 0.60 is unmet and no gain is predicted by the algebra.

## Why other QC routes remain deferred

An anonymous geometry-only polar-vector head obeys ordinary nuclear O(3), so planar reflection forces its normal component to zero and inversion symmetry can force the whole vector to zero. Electronic transition character can compensate those signs physically; a scalar state index does not carry that representation. Whole-vector phase-free losses, as used by [SchNarc](https://doi.org/10.1063/5.0021915), address label signs but cannot change that function class. A phase-blind tensor avoids this specific obstruction; the current PSD decoder already does so.

AO integrals S,D can be computed from geometry and a specified basis, but naively equivariant transition-density factors inherit the missing electronic-character problem. A neutral covariance over AO-pair coefficients can avoid assigning a signed state factor: each factor must satisfy tr(TS)=0, or covariance K must satisfy K vec(S)=0. T itself must not be constrained PSD. Coherent AO contraction remains more expensive and has many unidentifiable dark directions. Exact Gaussian TZVP basis conventions, integral parity and missing MO/X/Y/density supervision are unresolved; substituting def2-TZVP would be wrong. [PySCF's primary response source](https://pyscf.org/_modules/pyscf/tdscf/rhf.html) separates the AO operator integrals from MO/X/Y contractions and distinguishes full transition moments from X-only convenience NTO analysis. These ingredients do not establish useful transition-density labels here.

The learned Hamiltonian/operator idea has a primary precedent in [LUSH](https://arxiv.org/abs/2609.01871), but that paper does not validate this congruence or transfer to this QM9S benchmark. Our previous synthetic response audit already found sorting, degenerate-root and eigenvector-gradient failures. The present construction avoids that eigensolver and C-root gauge, at the cost of making a much narrower empirical claim. Round01/02's right-CG F, raw-M penalties and frozen readouts did not test a common A-dependent spatial transform. Their negative evidence still cautions against overallocating another small readout.

The primary arXiv abstract and PDF were opened successfully on 2026-09-30: **Latent unified smooth Hamiltonians for excited state chemistry**, David Juergens et al., arXiv:2609.01871v1, submitted 2026-09-01. Its abstract describes thymine and azobenzene examples. This citation supports only the existence of a shared latent-operator approach; no accuracy, symmetry guarantee or degeneracy remedy is transferred to the present design. The HTML rendering endpoint failed, while the primary abstract/PDF succeeded; the receipt distinguishes those outcomes.

## Brief independent Round05 fairness review

The selected control/F/decorrelation/both factorial is defensible as a bounded **joint-learning** experiment, not a QC-operator validation. Identical fresh base tensors, independent shared data order, fixed realized LR/objective, all labels, fixed60 reporting and no selective extension address the major confounds. Dormant adapter schema in controls is not active capacity matching; the proposal says so. The old .418119 score is contextual, not a comparator on v2. Seed11/60-epoch selection and the +.003 allocation gate do not establish significance; paired new seeds remain needed.

Preserve the corrected `HISTORICAL_NONREPETITION.md`: the old .373493 scratch MTO was a **direct-f head** with a raw-f objective, not the original PSD LE+Ls readout. That correction strengthens the actual distinction without erasing prior null results. I find no reason to reopen Round05's fixed settings. This note does not independently certify its pending implementation, data loader or split audit.
