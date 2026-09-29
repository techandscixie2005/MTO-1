# Independent label audit and deferred response-operator hypothesis

2026-09-29. Read-only source/schema/theory review; no new label extraction, test inference, fitting, or response-head implementation. Keep the four-arm F/orth pilot as the current priority.

## Audited labels and what reconstruction establishes

Server source `/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925/{README.md,dataset_audit.md,extract_qm9s.py}` contains geometry, atom identity, ordered excitation energies, printed length-gauge oscillator strengths, transition electric dipoles and derived A=μμᵀ, plus optional velocity/magnetic dipoles and calculation metadata. The extraction does not parse orbital coefficients, occupations, AO overlaps/dipole integrals, full X/Y response vectors, or transition-density matrices. Their availability is **not established** by the existence of excited-state labels. The raw Gaussian logs were not copied to this server dataset. The prior targeted audit could not access the historical E:\DATA source log on the current host.

All133727 successful calculations have ten states; the remaining158 source files have zero states, missing geometry, and failed termination/convergence checks. They are not new holdout data. The recorded route is `#p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read`; the targeted ID14562 record reports DoRPA=T. `nosymm`/printed `?Sym` do not supply reliable electronic irreducible-representation labels.

The full extraction audited f=(2/3)(E_eV/27.211386245988)|μ_AU|² against printed raw f: mean absolute discrepancy2.48655e-5, max7.49881e-5, consistent in scale with printed1e-4 strengths. This checks E/μ/f units and retained dipole consistency; it does **not** validate reconstruction of μ from amplitudes/densities, which are absent. Original f stays primary and every valid zero remains scored. Source parser promotes higher-precision electronic-transition rows2–4 only after component agreement within5.1e-5 with the labeled electric-dipole table; absent values use NaN/masks.

### Full-TDDFT/NTO requirements

For restricted real singlets, a length-dipole reconstruction must retain the response convention and both excitation/de-excitation contributions. PySCF's official implementation contracts its MO dipole integrals with X+Y (and X−Y for anti-Hermitian velocity operators), including its normalization-dependent factor2; oscillator strengths use2ω|μ|²/3 with ω in Hartree. Its convenience `get_nto` instead normalizes X and ignores Y. Those NTOs cannot simply substitute for exact full-TDDFT response reconstruction. [PySCF implementation](https://pyscf.org/_modules/pyscf/tdscf/rhf.html).

NTOs compress a chosen transition matrix via SVD into paired occupied/virtual orbitals. Retaining only electron/hole populations discards phase information needed for coherent reconstruction; paired factors and singular amplitudes must be retained consistently. [Q-Chem NTO documentation](https://manual.q-chem.com/5.2/Ch7.S13.SS2.html).

A future label audit must first prove complete coefficient support (not only thresholded printed excitations), spin/normalization conventions, X and Y availability, basis/overlap and AO/MO transforms, coordinate frame and units, then reproduce μ and raw-f rounding before using any auxiliary label. Signs of orbitals and excited states are gauges. Within degenerate occupied/virtual or excited subspaces, compare projectors/subspaces or use a consistent covariant transformation; arbitrary individual vectors cannot be treated as unique targets. SVD-degenerate NTO pairs have the same ambiguity. Do not infer transition densities from density differences or independently squared electron/hole marginals.

## What the existing MTO head already represents

`experiments/qm9s_eta_Ef_20260926/frozen_reference/models_ea.py` forms Q from a gated sum of tensor channels, then C=βI/√3+Q and A=CCᵀ. Since Q is traceless, trace(A)=β²+||Q||². The square of the channel sum already includes coherent **within-state, cross-channel** interference. Calling the current PSD decoder incoherent would be wrong. The scalar β alone can represent arbitrary nonnegative invariant strength; no magnitude expressivity ceiling follows from PSD.

The untested hypothesis is a useful **cross-state** shared response structure. Current per-state spatial A tensors do not determine off-diagonal overlaps of signed transition amplitudes between different states. Previous frozen within-state Gram regression did not test that structure. Previous P-only polar-vector/rank-one experiments simultaneously changed bypass, CG outputs and readout and therefore do not settle this hypothesis.

## Deferred geometry-only construction

Let a geometry encoder produce a shared real symmetric latent state-space H and a PSD state-space Q=BBᵀ. Let H=U diag(E) Uᵀ. Predict ground-to-excited strengths as s_k=[Uᵀ Q U]_kk and f_k=(2/3)(E_k/27.211386245988)s_k. All ten target states remain scored in the original energy order; strengths follow the same U. Positive excitation energies require an explicit shared shift or positive-spectrum parameterization, frozen before fitting. No QC amplitudes, orbitals or integrals are inputs at deployment.

Q corresponds to a physical real dipole Gram only when Q_ab=μ_0a·μ_0b and rank(Q)≤3. An unrestricted latent PSD Q is instead a flexible strength surrogate. Either can be learned as an inductive bias; E/f alone identify neither off-diagonals nor a unique latent basis and do not establish physical wavefunctions, NACs, or a dipole operator. This is an empirical response-head hypothesis, not a claim of electronic reconstruction.

Under a common orthogonal latent-basis change V, H→VᵀHV and Q→VᵀQV leave predictions unchanged. Eigenvector signs cancel. At exact excited-state degeneracy, arbitrary subspace rotations redistribute individual strengths; only subspace total is invariant when energies coincide. Near degeneracy, eigensolver derivatives require explicit treatment and validation. Ordinary benchmark ordering stays unchanged; oracle swaps/grouping must never replace raw-f accuracy.

An ordinary permutation-invariant O(3) polar-vector output is forced to zero on an inversion-symmetric nuclear geometry: inversion maps to an identical-atom permutation yet negates the vector. Physical opposite-parity electronic transitions may be bright because the state phase/representation also transforms. A scalar query index does not encode that electronic gauge. A phase-free spatial tensor or state-space Gram can retain nonzero intensity; merely adding a sign-invariant vector loss cannot cure a vector architecture's forced zero.

This construction and caveats also appear in the prior independently reviewed `research/oscillator_r2_20260928/lush_feasibility/FEASIBILITY_REVIEW.md`. Its shared-eigenbasis motivation comes from [Juergens et al., LUSH primary paper](https://arxiv.org/pdf/2609.01871v1); that work's excited-state split across nine QeMFi species is different from the present connectivity-held-out QM9S task. The current audit does not infer transfer performance.

## Prerequisite tests before any response pilot

1. Two-state analytic H=EcI+xσz+yσx and non-diagonal PSD Q: reproduce exact strengths and signed-interference changes; compare a diagonal-Q ablation.
2. Common basis rotations and eigenvector sign flips preserve outputs to FP64 tolerance; test gradients by finite differences away from degeneracies.
3. Approach zero gaps from fixed paths. At zero gap, verify invariant subspace totals and document undefined individual-state assignment; no silent gap clamping/jitter.
4. Centrosymmetric bright-transition synthetic example: permit nonzero scalar/tensor intensity without requiring a unique oriented μ.
5. If physical rank3 Gram is chosen, verify PSD/rank and show that three latent columns are a gauge, not audited laboratory polarization. Retain all ten benchmark states; no disposable upper-state buffers.

These tests are a proposed prerequisite only. The current factorial must inform whether this more expensive and less identifiable head is the next priority. No implementation or training is authorized by this document itself.
