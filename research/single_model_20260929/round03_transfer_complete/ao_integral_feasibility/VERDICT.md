# AO integral feasibility: useful ingredient, unverified learned readout

**Verdict:** geometry and an explicit fixed Gaussian basis suffice to compute AO overlap and electric-dipole integrals. MO coefficients and TDDFT X/Y amplitudes are not required for those integrals. An AO-pair covariance can include onsite transitions missing from point-charge models. It remains an empirical architecture hypothesis; neither the density nor accuracy improvement follows from the integrals. No architecture, QC calculation, package installation or fit was performed.

## What is available

The existing extraction manifest records Gaussian16 B.01, route `#p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read`, DoRPA=T for133,727 successful neutral-singlet records, positions in Å and transition moments in atomic units. One bounded metadata sample prints `TZVP (5D, 7F)`. This identifies the recorded keyword and sampled angular convention. Full basis exponents/contractions, AO ordering/normalization and cross-program integral parity are not audited. Do not silently substitute `def2-TZVP`. `geom=check` does not identify the prior geometry-optimization method.

The pinned runtime has no PySCF; nothing was installed. Its official integral API and [AO integral example](https://github.com/pyscf/pyscf/blob/master/examples/gto/20-ao_integrals.py) expose `int1e_ovlp` and `int1e_r` from molecular geometry/basis and a chosen origin, without running SCF. The [libcint paper](https://arxiv.org/abs/1412.0649) describes analytic Gaussian integrals. Their availability does not provide transition coefficients: [PySCF's response implementation](https://pyscf.org/_modules/pyscf/tdscf/rhf.html) separately contracts AO operators with occupied/virtual MO coefficients and X+Y for length moments. Gaussian conventions still require independent matching; no X-only NTO or exact reconstruction claim is warranted.

## Algebra and constraints

For real AO functions, let `S_ij=<chi_i|chi_j>` and `D^a_ij=<chi_i|r_a|chi_j>`. Represent an effective real length-transition coefficient matrix by T, with `mu_a=tr(T D^a)`. The symmetric part suffices for this contraction with symmetric D; it does not reconstruct the complete response density or velocity operator.

Let t be the flattened T in a consistently normalized pair basis, d_a the flattened D, and s the flattened S. A PSD covariance K over coefficient space gives `A_ab=d_a^T K d_b`. The condition `K s=0` makes it origin independent because moving the dipole origin by c gives `d_a -> d_a-c_a s`. A factorization `K=sum_l t_l t_l^T` is sufficient when each factor has `tr(T_l S)=0`.

**PSD applies to K, not T.** A nonzero transition matrix need not be PSD. Requiring T PSD and `tr(T S)=0` with positive-definite S would force T=0. A general covariance may yield rank2/3 A even when a single real transition has rank1 `mu mu^T`; it is then an empirical response covariance, not an identified pure-state transition density.

Under an invertible AO basis change `chi'=chi B`, the correct laws are `S'=B^T S B`, `D'=B^T D B` and `T'=B^-1 T B^-T`. This preserves moment contractions and overlap neutrality. Plain `tr(T')=0` in a nonorthogonal basis is insufficient. Rebuilding an ordinary Euclidean projector in each arbitrarily transformed basis is also not equivalent to transporting K. AO permutation and O(3) covariance require the corresponding AO-shell representation law, including parity; a generic flat matrix network does not acquire it automatically.

One possible mathematical convention is symmetric orthogonalization by S^-1/2 followed by traceless coefficient space. That is a unique matrix function on positive-definite S and can be computed as a detached geometry feature. It avoids training through a learned electronic eigensystem. Near linear dependencies still make it ill-conditioned, and any rank truncation needs an explicit policy. This is a convention to audit, not an implemented design.

## Bounded CPU evidence

`synthetic_audit.py` uses only NumPy and analytic normalized primitive s/p Gaussians at one center. These are not the target TZVP basis and contain no dataset examples.

| Check | Result |
|---|---|
| Onsite s–p_z dipole, exponent0.8 bohr^-2 | mu_z=.55901699, A_zz=.3125 despite zero nuclear-coordinate span |
| PSD covariance, overlap neutrality, origin shift | Minimum eigenvalue0; K vec(S)=0; A unchanged exactly |
| Proper/improper Cartesian transformation with consistent AO transformation | Maximum A error3.34e-16 |
| AO permutation / electronic sign / inversion | A or covariance unchanged exactly; odd T can have even nonzero K |
| Correct nonorthogonal basis transport | A error2.78e-16 |
| Naive ordinary-trace neutrality | Residual overlap charge.68; origin shift changes A by1.4847 |
| Recomputed Euclidean neutral projector | Basis-dependent A change1.3798 |
| Coherent pair covariance | Opposite pair terms cancel to0; deleting cross terms gives strength2 |
| Add a neutral dark AO component | K changes by Frobenius norm2.4495; A unchanged exactly |
| Rotate two exactly degenerate electronic transitions | Individual A changes.15625; group sum preserved to5.56e-17 |

The onsite result removes the specific nuclear-span obstruction. It does not mean an isolated spherical atom can choose a unique directional transition from geometry: the three-component aggregate is isotropic. A covariance remains phase blind, and exact-degenerate individual state tensors retain their electronic basis ambiguity. E/f, or even A, cannot identify the many dark directions of K.

## Design verdict and smallest useful hypothesis

**Go:** retain explicit AO integrals as a possible geometry-only ingredient. They remove the proven nuclear-span limitation through onsite and pair matrix elements. **No-go:** do not start an AO architecture fit or claim that this fixes false-bright errors. The existing direct C C^T head already represents arbitrary PSD output tensors; AO structure would add an inductive bias, not establish a missing output range. An unconstrained learned K can still produce excessive strengths. There is no empirical evidence here that it reduces those errors.

A narrow future question would ask whether onsite/pair structure improves an empirical, neutral, equivariant response covariance compared with a matched direct PSD tensor readout. Current data do not justify a physical transition-density, Hamiltonian, NTO or sum-rule interpretation. Complete X/Y/MO information remains missing.

Dense K costs O(M^4): even an illustrative M=100 symmetric AO space has5,050 pair coordinates and needs204MB of FP64 covariance per state. Low-rank factors cost O(r M^2), but must preserve AO symmetry, neutrality and coherent terms. Their channel mixing remains a gauge freedom. A zero-initialized factor has zero quadratic gradient; an additive PSD residual can only increase trace and cannot correct excessive strengths. Any later baseline-preserving insertion must address those constraints explicitly. No such insertion is proposed or approved here.

The smallest justified next preflight, only if separately authorized, is an integral-only test on a few analytic/synthetic geometries: pin explicit exponents/contractions and spherical/cartesian ordering, then verify overlap/dipole values, units, origin shift, proper/improper rotation and AO permutation against a trusted integral implementation. Include onsite s–p and near-dependent overlap cases. It needs no target labels, SCF, TDDFT, training or large cache. PySCF is currently absent; this note authorizes no installation. Only after that ingredient is verified would a scoped architecture proposal be concrete enough to review.

### Geometry/vector frame provenance

The extractor copies electric/velocity components without a rotation and selects the last Standard orientation if present, otherwise the pre-label Input table. Successful unambiguous records require one such geometry table. The manifest reports Input coordinates for every complete record and nosymm in every route. The historical108-log spotcheck separately verified copied Input coordinates and electric components; it did not test their rotation relationship or velocity frame. Downstream preparation additionally asserts that curated positions and A equal the extracted values after FP32 conversion, and the dataset loader applies no rotation.

This supports a shared Input-frame convention and reveals no mismatch. A direct vector-to-geometry alignment test remains absent; scalar f reconstruction cannot provide it. Do not infer a mismatch from eta performance. Current eta0 LE+Ls uses trace(A), unchanged by a pure orthogonal frame change, whereas future directional tensor/vector supervision needs stronger alignment evidence. See `FRAME_PROVENANCE.json` for code paths and hashes. The fixed round03 fit is untouched.
