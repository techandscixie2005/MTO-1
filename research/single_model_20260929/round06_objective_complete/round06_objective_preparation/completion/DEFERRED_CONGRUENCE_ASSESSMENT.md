# Deferred shared-PSD congruence after Round06

Analysis only. Root closes the Round06 objective pair under its frozen gate. No implementation, technical updates, inference or fit is authorized by this note. A concrete proposal may follow only after completed-round publication; preparation needs a new decision.

## Why it remains a distinct, bounded question

Round06 did not improve the retained v2 R².447169 reference. Direct raw-f loss reduced selected false-bright SSE versus its weak contemporaneous control, but worsened true bright-tail, MAE and energy errors. Both trajectories continued fitting TRAIN after their validation-selected checkpoints. This does not identify objective mismatch, a physical mechanism or a generalization cause. The control rerun difference is substantial and has known execution-workload differences; one seed cannot establish robustness.

The existing deferred proposal changes the dependence of each PSD output on the other states' latent Cartesian tensors while keeping original LE+Ls. It does not repeat the right-F adapter, raw-M penalty, scalar calibration or this raw-f objective. The original decoder already has coherent channel interactions and spans PSD outputs; the claim is a different shared inductive bias, not missing PSD expressivity or a discovered quantum response operator.

Keep the exact already audited design as the starting proposal: R=ΣA_k, S=tr(R), epsilon=1e-6 in squared dipole atomic units, B=(R+epsilon I/3)/(S+epsilon), q=sqrt(tr(B²)/3). One177-parameter gate Linear(9,16)–SiLU–Linear(16,1) produces b=.25 tanh(g). The same nine invariant inputs are five log1p element counts, mean predicted E in Hartree, log1p S, log1p Σ(E_H trA), and q. These use only geometry/predictions, not QC targets. Last gate layer starts at zero, with exact common gate/base tensors and isolated data-order RNG.

Three fresh arms would be required:

1. Original reference: A'_k=A_k.
2. Scalar control: A'_k=(1+bq)² A_k.
3. Tensor candidate: A'_k=(I+bB) A_k (I+bB)ᵀ.

The two modified arms match gate inputs, parameter count, identity initialization and Frobenius norm of the matrix perturbation. Their difference tests directional context beyond a molecular scalar rescale; it does not equalize every gain range or expressive property. A contemporaneous original control is essential given observed repeat variation. Allocation should require selected tensor R² at least+.003 above BOTH modified scalar and original controls AND the retained .44716940136585204 reference, with full state/tail/energy reporting. Original LE+Ls, common fresh initialization/orders/optimizer/fixed60 budget and validation-only selection should remain fixed initially. No re-use of trained Round05/06 parameters, unchanged raw-f continuation or blanket search is justified.

## Mathematical and scientific limits

The transform uses A rather than arbitrary C factors, avoiding per-state phase/factor-gauge choices. It preserves PSD and O(3) covariance, including reflections/inversion; it inherits atom-permutation/translation invariance. The common L gives the appropriate exact-degenerate block-sum property for explicit dipole fixtures. It does not recover individual electronic characters or justify permuting ordered benchmark labels. Gate inputs are defined because the actual base uses softplus E and PSD A. Eigenvalues of L stay between.75 and1.25; no eigensolver is used. The scalar and tensor arms coincide in the isotropic case and have live first derivatives for generic nonzero PSD outputs.

A directions under trace-only supervision are latent features, not identified physical transition polarizations. This module cannot by itself revive exactly zero A, change its rank or arbitrarily transfer strength between states: a shared sign of b changes strengths in the same direction, with different magnitudes. It may therefore leave the observed tail tradeoff unresolved. Fixed epsilon and differentiable aggregate features need FP32/autograd conditioning checks near zero; the old NumPy algebra PASS does not establish production integration, gradient or inference correctness.

The preserved design and primary-source discussion are in `QC_DESIGN_RECOMMENDATION_20260930.md` (SHAc503e4d91c3e90af1de3702648e2785195a9e28d95177f8ac0838472fd4abcc8) and `qc_response_congruence/`. Before any fitting, a new reviewed preparation would need identity/O(3)/factor-gauge/isotropy/degenerate-block tests, live gate/base gradients, exact parameter controls and fresh initialization, a bounded TRAIN-only resume/export fixture and fail-closed publication/authorization gates. Geometry-only one-checkpoint inference remains mandatory.

This is the smallest presently specified structural follow-up with a materially different rationale. It is not a prediction of reaching0.60. More expensive AO/density/electron–hole approaches still require the missing electronic-character/supervision/convention audits; no anonymous polar-vector or naive AO-density head should be introduced as a shortcut.
