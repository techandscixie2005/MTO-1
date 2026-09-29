# Response-adapter feasibility: reject a direct eigensolver now

**Verdict:** an independently learned symmetric H plus PSD strength Gram Q is mathematically meaningful, but is not a safe identity-preserving addition to the present state-wise predictor. The CPU audit found concrete sorting, degeneracy and gradient failures. A smaller explicit state rotation of existing tensor factors is numerically viable as an empirical adapter. It still has unresolved representation-gauge dependence and cannot fix overall transition-strength scale. Do not launch it before the current factorial is analyzed.

This audit used only synthetic FP64 matrices. No molecular data, labels, checkpoints, model inference, fitting or GPU APIs were used. Files stayed under local `research_state/response_feasibility/`; the script ran through SSH stdin with CUDA visibility empty and one CPU thread. The frozen pilot was unchanged.

## Findings from executable examples

| Question | CPU result | Consequence |
|---|---|---|
| Common latent basis and eigenvector sign | Joint H/Q rotations and permutations preserve f to1.8e-14; eigenvector signs cancel exactly. | The proposed shared-basis algebra is sound away from degeneracy. |
| Identity at zero adapter | Base E=[2,1,3] and f=[.2667,.6,1] become E=[1,2,3], f=[.6,.2667,1] under eigh(diag(E)). | Ascending sorting changes original state-slot outputs even with no learned correction. Sorting strengths separately is invalid; matching eigenvectors back to anchors adds discontinuous assignment. |
| Exact degeneracy | H=2I and Q=[[1,.8],[.8,1]] permit strengths[1,1] or[1.8,.2] under equally valid eigenbases. | Individual strengths are not basis-invariant. The equal-energy subspace total remains invariant. A sign-invariant loss does not resolve arbitrary subspace rotations. |
| Near-degenerate derivatives | With offdiagonal perturbation t, ds_low/dt=−1.6/gap:−16 at gap.1, about−1.6e9 at1e-9; NaN at zero. Finite differences agree at gap1e-3 to2e-10 relative error. | This is a genuine conditioning/singular-target issue, not merely an implementation bug. Silently clamping gaps would change the derivative without defining the missing target. |
| Conservation | A rotation preserves sum strength=5, while energy-weighted sum f changes from4.6667 to6.6667. | Unweighted strength conservation is not f conservation or a finite-state TRK guarantee. |
| Identifiability | Two positive Q matrices with identical diagonals have identical original E/f, yet their mixed strengths differ by.8. | E/f cannot identify cross-state operator elements. Many common H/Q basis choices also yield the same observables. |
| Identity rotation initialization | For diagonal Q, the rotation gradient is exactly zero; for a coherent Q it is−1.6. | A zero rotation plus diagonal strength matrix is a dead first-order initialization. Adding random offdiagonals merely chooses an unobserved prior. |
| Centrosymmetry | An even C tensor has nonzero strength under inversion; its Gram is O(3)-invariant to3.6e-15. | It avoids the forced zero of an ordinary permutation-invariant polar-vector head on an inversion-symmetric nuclear geometry. It is not thereby a physical transition dipole. |

The sorted eigenvalue and near-gap warnings match [PyTorch's primary eigh documentation](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html). The observed matrix derivatives above were independently computed, rather than copied from a paper.

## Minimal optional empirical variant

The current decoder already forms C_k=β_k I/√3+Q_k and A_k=C_kC_kᵀ. Its Q_k is a **coherent sum of within-state tensor channels** before squaring. That existing interference must not be described as absent.

Expose these existing C_k internally and flatten their entries into factor rows B_k. Their Frobenius norm squared is the original strength s_k=trace(A_k), so the state-space Gram G=BBᵀ has exactly the required diagonal. This is a rank≤6 Gram for symmetric3×3 C tensors, not a physical rank≤3 dipole Gram.

Use a shared ordered-pair network on invariant geometry/state features to produce a_ij. Define

    r_ij = G_ij / sqrt((G_ii+eps²)(G_jj+eps²))
    K_ij = kappa * r_ij * tanh(a_ij-a_ji)
    U = exp(K),  B_new = Uᵀ B
    s_new,k = ||B_new,k||²
    f_new,k = (2/3)*(E_k/27.211386245988)*s_new,k

K is skew-symmetric; U is orthogonal. Zero-initialize the pair network's last weights so K=0 and retain original output slots and energies. A final scalar bias cancels in a_ij−a_ji and should not be counted as useful capacity. Output C_new by the same state mixing, then A_new=C_new C_newᵀ if tensor outputs are required; the f–E–trace relation is retained. No sorting or eigendecomposition is performed. Formally one can define H=U diag(E) Uᵀ afterward, but this does not independently learn a physical Hamiltonian.

The correlation factor makes K covariant to independent row-sign changes B→SB: K→SKS and U→SUS, leaving strengths unchanged. Omitting that factor produced a.2381 change in the synthetic strength under an otherwise invisible row-sign flip. The tested construction preserved those signs and state permutations exactly, retained nonzero initial gate gradient, and had finite gradients at zero energy gap. Orthogonality and total-strength conservation errors were about1e-12. Zero factor rows and epsilon denominators remained finite.

The prototype does not implement a geometry encoder or pair network. It tests only this algebra with synthetic logits. `kappa=.1` and `eps=1e-8` are synthetic constants, not proposed validated training hyperparameters. A production proposal would separately freeze values, initialization, capacity controls and losses before fitting.

## Why this variant remains limited

1. **More factor gauges remain.** Two symmetric C square roots can produce exactly the same A but different cross-state Grams. Changing one principal-value sign preserved every source A while changing adapter strength by.1410 in the test. Independent whole-row sign covariance does not eliminate this larger ambiguity. Treat C as learned latent features, not reconstructed amplitudes.
2. **General degenerate-subspace covariance is not automatic.** A shared ordered-pair MLP has no defined transformation under an arbitrary rotation of electronic-state features. In a three-state example with outside-state mixing, rotating a designated degenerate subspace while keeping pair logits fixed changed that subspace's total strength by.000625. Thus this proposal does not solve physical degeneracy gauge, even though it avoids eigengrad blowup. A common-basis-covariant pair operator would need an explicit representation law and further tests.
3. **Pure rotation cannot correct total strength.** With all true strengths zero and two equal-energy predictions summing to5, the squared strength error is bounded below by5²/2=12.5. Redistribution can spread an amplitude error but cannot make the molecule dark. This limits relevance to the previously observed all-dark-label concentration; no sample should be excluded.
4. **It can still be inactive.** If offdiagonal G is zero, the sign-covariant rotation gate is zero. This protects a special orthogonal configuration but limits correction. A diagonal Q plus zero rotation has the stronger universal zero-gradient problem shown above.
5. **The latent dimension is six, not an audited electron-hole basis.** Neither this factor nor an unrestricted learned H/Q identifies transition densities, orbital phases, excited-to-excited operators or nonadiabatic couplings. Geometry-only inference is feasible, but physical interpretation remains a hypothesis.

A smooth spectral matrix function such as a broadened response can avoid resolving individual vectors inside exact degeneracies. It would not identify the benchmark's individual ordered raw-f labels; it must not replace raw-f evaluation or silently merge states.

## Conditional next decision

First complete the frozen F/raw-state-decorrelation factorial and its required independent-seed decision. If evidence later motivates cross-state intensity allocation, the explicit rotation above is a more controlled first probe than a learned eigensolver. Preserve all existing inputs/readout paths and exact initialization; compare against a matched unchanged model and a simple capacity control. Track total-strength error separately from state redistribution and bright-tail error. Require actual MTO C-factor identity, spatial/permutation/sign tests and resume tests before any run. Do not claim physical shared response without resolving its gauge limitations.

The broad shared-eigenbasis motivation has a primary precedent in [Juergens et al., Latent unified smooth Hamiltonians](https://arxiv.org/abs/2609.01871), but that paper does not establish transfer gains for this MTO/QM9S split. A future physical density route still needs the missing full-TDDFT amplitude/integral audit: [PySCF's official response source](https://pyscf.org/_modules/pyscf/tdscf/rhf.html) distinguishes X+Y transition moments from its X-only convenience NTO analysis. Current E/μ/f consistency cannot substitute for such a reconstruction test.

## Reproduce

From the local task directory, pipe `synthetic_audit.py` to the existing server environment with `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; capture stdout as `SYNTHETIC_RESULTS.json`. The script asserts explicit CPU isolation and imports only NumPy/Torch. It does not create server files. All results and failures above are in that JSON; no source credentials, arrays from molecular data, or model weights are present.

No architecture or experiment is authorized by this feasibility note. Parent/root decides after factorial evidence.
