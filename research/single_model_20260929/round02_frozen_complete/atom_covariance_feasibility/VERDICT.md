# Atom-centered transition-response covariance: bounded feasibility audit

**Reject `A=Rᵀ C K C R` as a general replacement for the dipole tensor.** It is a valid PSD, neutral, geometry-only empirical construction when K obeys the required symmetry and positivity, and an atom-indexed kernel avoids one centrosymmetry failure. It still omits dipole directions outside the span of the nuclei. No model implementation, label-projection audit or training is authorized by this note. Round02 remains the sole fit.

## What works mathematically

R has one Cartesian position per atom; `C=I−11ᵀ/n` acts only on valid atoms. If K is PSD and permutation-equivariant, `P=C K C` is PSD and `P1=0`. Therefore translation drops out, atom permutations leave A unchanged, and rotations/reflections transform A as `A′=O A Oᵀ`. The CPU toy checks pass within1.3e-14. Equal-weight centering is sufficient for charge neutrality; it is not an AO overlap metric or a wavefunction orthogonalizer.

For a centrosymmetric identical-atom dimer at x=±1, `K=I` gives `tr(A)=2`; the Gaussian distance kernel gives1.72933. Its atom-indexed columns transform with both indices under permutation, allowing nonzero covariance despite the absence of a uniquely signed equivariant transition vector. In contrast, fixed-rank scalar node features identical on inversion partners give `Rᵀ C B=0` and `A=0`: replacing the kernel with a conventional invariant node-factor Gram matrix reintroduces the collapse.

Off-diagonal P entries retain coherent pair contributions. A neutral charge toy q=(1,−2,1) at x=(−1,0,1) has diagonal strength2 canceled by cross terms−2. Summing only atomwise squares would destroy that cancellation. A factor Gram matrix is one learned head, but its latent-factor sum of squares does not identify physical transition amplitudes or cross-state coherence.

## Restriction that defeats a general replacement

`rank(A)≤min(3,rank(CR),rank(P))`, and A annihilates every direction perpendicular to centered nuclear positions. Consequently:

- Every exactly planar geometry has zero predicted out-of-plane dipole strength.
- Every linear geometry has zero perpendicular strength.
- A one-atom geometry has zero strength in every direction.

These restrictions are stronger than the symmetry constraints on a transition dipole *tensor*. For a molecule fixed under reflection through its plane, the toy target `A=diag(0,0,1)` is itself reflection-invariant, yet the proposed head cannot produce it. Electronic transition densities can have on-site spatial structure that nuclear point charges cannot represent; the nuclear-span failure is our algebraic deduction, independent of any claim that these examples dominate the current dataset. No dataset prevalence was measured.

The limitation is consistent with the use of atomic transition dipoles and quadrupoles in addition to charges in [Fujimoto's TrESP-CDQ research](https://doi.org/10.1063/1.4902758). That work concerns electronic couplings from quantum-chemical transition densities; it does not validate the proposed learned P. Similarly, [Veit et al.](https://arxiv.org/abs/2003.12437) distinguish charge motion and local atomic polarization for molecular permanent dipoles; their ground-state result is contextual evidence, not proof about MTO transition labels.

## Positivity, gauge and optimization caveats

- Symmetric positive pair entries do not ensure PSD. The three-atom counterexample has minimum eigenvalue−.2238, and its centered P remains indefinite (−.1667).
- A Euclidean distance Gaussian with one positive bandwidth is PSD. A general atom-indexed pair factor B also permits `K=B Bᵀ`, evaluating `A=(Bᵀ C R)ᵀ(Bᵀ C R)` without eigendecomposition. The toy derivative is finite at repeated eigenvalues. This is an algebraic option, not a proposed implementation or solution to the nuclear-span restriction.
- `B→B O` leaves P unchanged. More seriously, adding a neutral `q qᵀ` with `Rᵀq=0` changes P while leaving A unchanged. The toy changes P by Frobenius norm6 with exactly unchanged A. Even A labels cannot identify P; E/f labels provide less information. No X/Y, NTO, AO-integral or physical transition-density reconstruction follows.
- Neutrality does not enforce rank-one pure-transition A, quantum-state orthogonality, state phase conventions, degeneracy handling or an oscillator-strength sum rule. A real isolated transition has a dipole outer product; a general PSD covariance is an empirical relaxation. Positive energy and PSD A ensure nonnegative f, but not conservation across states.
- Coordinates in Å require a declared conversion/scale if A is interpreted in atomic units. Learned atom covariances are not explicit AO dipole integrals.
- A naive positive-semidefinite addition only raises `tr(A)` and cannot directly correct excess oscillator strength at fixed E. Zero-initialized squared factors preserve identity but have zero first derivative, confirmed by the toy. A future insertion would have to permit attenuation, preserve PSD/O(3), and have nonzero useful gradients; this audit does not invent an insertion to rescue the proposal.

## Available features and smallest remaining hypothesis

The pinned `frozen_reference/upstream/models.py` exposes atomic scalar features s and equivariant tensors t before MTO pooling. `invariants(h)` provides scalar0e and squared1o/2e norms; atom identity, pair distances and learned state queries are also available. Thus a geometry-only state-conditioned pair operator is computationally possible without QC inputs. Those features do not remove the point-charge projection restriction, and a compact fixed-column invariant factor can erase the needed inversion-odd atom-index modes. The current M-only cache does not retain atomwise features and cannot simply be reused for this head.

At most, keep a partial empirical spatial prior as a deferred hypothesis. A future investigation would first need a separately authorized TRAIN-only geometric-span/label audit and a justified role beside, rather than as, a complete dipole representation. No such audit, on-site fix, new objective or fit is selected here. Current round02 outcomes take priority.

## Reproduction and evidence

`synthetic_audit.py` uses only small synthetic arrays, NumPy and CPU PyTorch; no model, data, labels or GPU are loaded. `SYNTHETIC_RESULTS.json` contains the numerical checks and counterexamples. The manifest records source hashes and the read-only backbone evidence paths. All files are lightweight and belong in a later archive, outside frozen round01/round02 source closures.
