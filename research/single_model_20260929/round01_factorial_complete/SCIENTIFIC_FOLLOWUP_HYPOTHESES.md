# Follow-up hypotheses recorded before pilot results

## Transition-dipole phase and molecular symmetry

A geometry-only, atom-permutation-invariant equivariant polar-vector predictor with invariant state queries obeys mu(RX)=R mu(X). If RX is the same molecular geometry after an allowed atom permutation, its output must also satisfy mu(X)=R mu(X). At an inversion-symmetric geometry this forces that predicted vector to zero. A physical transition dipole between different electronic symmetry states can remain nonzero because the excited-state wavefunction contributes its own phase/symmetry transformation. Therefore a plain polar-vector head can impose stronger constraints than the physical transition property.

This is a mathematical architectural concern, not a demonstrated explanation of any historical MTO error. A=mu mu-transpose removes an overall sign and is even under inversion; the current PSD head avoids that particular zero-vector restriction. For degenerate transitions, the relevant ambiguity also includes basis rotations inside a state subspace, so individual vector or tensor targets need an explicit convention or an invariant subspace target. A phase-insensitive loss addresses arbitrary signs in targets; it does not by itself remove a representational zero forced by the input/output symmetry class.

[The original SchNarc paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.0c00527) establishes the need to handle arbitrary electronic wavefunction phases when learning interstate properties and introduces phase-free training. The point-group constraint above is our deduction from equivariance, and must be tested on simple symmetric geometries before a new polar readout is proposed.

## New-rationale requirements for later experiments

- A renewed coherent polar readout must address state phase/symmetry, retain useful invariant skip information, or add independently audited physical supervision. The previous POuter campaign already summed signed1o channel contributions before taking an outer product.
- An NTO-style factorization must preserve signed coherent contraction through dipole integrals. Squaring weights or averaging per-pair intensities before summing can remove interference.
- A shared learned response operator can be tested with geometry-only inputs as a statistical architecture. Current labels cannot validate that its factors are literal electron-hole amplitudes.
- Explicit AO dipole-integral reconstruction needs an audited basis, overlap metric, MO coefficients and fullTDDFT normalization/X/Y data. These are absent from the available extraction. New QC calculations may enable a separate train-only audit, but inference cannot require unavailable QC labels.
- If the right adapter succeeds, distinguish additional capacity, nonlinear gating, and state sharing using separately controlled follow-ups. If it fails, do not conclude all effective transition operators are unhelpful; the present map changes one specified right operand while preserving all skips and output constraints.
