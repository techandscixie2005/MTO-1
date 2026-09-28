# Independent implementation review: PASS

Reviewed the fixed two-arm Gram implementation and saved train-only preflight, then independently verified all15 source and8 code hashes against the refreshed handoff. No full fitting, model inference, test access or launch was performed by this reviewer.

The implementation forms the Cartesian Gram in FP64 after promoting frozen FP32 tensors/basis, uses all528 upper coordinates with sqrt2 off-diagonal factors, and shares the same129 h/base features and normalization across arms. Stable centered moments, mean-MSE ridge lambda=.001, unpenalized intercept, Cholesky, normal-equation residual and nested regularized-objective gates match the protocol. The max(0,signed) projection is applied after fitting and audited separately.

Preflight covers first64 training molecules for cache/rotation/permutation/trace checks and first16 for ridge. CPU versus CUDA moments differ by at most4.44e-16; both coefficient vectors agree. Source rotation max difference1.09e-5 and permutation1.66e-6 pass the declared FP32 combined tolerances. No claim of exact finite-precision invariance is made.

Two review findings were resolved before sealing: direct helper/config/bootstrap/identity/comparator files are now pinned, and the CUDA moment path used by the full run is tested against CPU/dense moments. The preflight description was corrected accordingly.

Source hashes and coefficient/statistics hashes enforce train-fit freeze before validation Gram generation/prediction; comparisons retain exact validation IDs/raw labels. Full-array storage decoding is not itself evidence of test leakage; fitting/evaluation consume train/validation rows only. Whole-molecule and connectivity-group bootstrap logic retains all states and recomputes pooled SST. Shared GPU lock, idle/health/UUID/ECC gates and feature progress manifests were inspected. A safe abort may require reviewed recovery; no automatic change of scientific settings is permitted.

PASS applies only to the exact hashes in IMPLEMENTATION_REVIEW.json, handoff114723976673b6598573774e28608ef3961426e242cb28ee76da6be6016a4e0f and preflight4ecd34d0309d46c748c20c4857e677b0fbfef4bc5b4104c2b9a4ed9fbe8a6fe0. Root has standing authorization for one gated run; executor must recheck admission and stop on material mismatch.

Interpretation limits remain: fixed ridge is not an optimum search; added features also add coefficients; a negative linear probe cannot establish a scalar-invariant readout or backbone ceiling; reused validation is exploratory. No promotion, test access or result-driven refit is authorized.
