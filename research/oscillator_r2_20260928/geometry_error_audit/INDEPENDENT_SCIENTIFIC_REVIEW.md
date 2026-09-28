# Independent scientific review: geometry and validation error

PASS for the saved descriptive audit. Report SHA bc97cdb2c1c949cf53eed6c5ab0610c1254b9f4405e37a28a5707affdbdc4d70; metrics SHA b088b23a5098149c5a127623a5a9315f6d4c979a501eeb3fd0af46285eaccbbd. This additive review leaves the original report, arrays, evaluation population and active studies unchanged.

## Independent checks

Recomputed train/validation geometry with a separate vectorized covariance/eigenvalue calculation from non-padding coordinates, after inspecting the audit reader. Coordinates are converted to FP64 before unweighted centering; covariance is X.T X/n_atoms, with eigenvalues descending. q2 and q3 are lambda2/lambda1 and lambda3/lambda1. These are dimensionless shape ratios; they are not mass-weighted inertia eigenvalues. Train eigenvalue minima were nonnegative, and no degenerate covariance appeared. The train-only q2 quantiles reproduce 0.05941956, 0.09867099, 0.21033972 and 0.45009813; the extra near-linear cutoff is fixed at 1e-5. The right-closed bins and all train/validation bin counts reproduce.

Verified input hashes, IDs/global indices, raw FP64 truth and masks for all three saved validation predictions; independently recomputed each stratum's SSE, median/P90 molecule SSE, statewise signed-error means, overprediction fractions and zero-label counts. Equal3 is exactly the fixed arithmetic mean. Rechecked selected train/validation connectivity metadata against IDs. No model inference, fitting or test-row analysis was performed.

## Evidence and interpretation

- Full retained validation: eta0 SSE 94.2051209294 and R2 0.40529411834; equal3 SSE 87.2150742627 and R2 0.44942146327.
- ID14562 has q2 1.58623969722e-6, q3 2.21841820540e-10, composition C8H2 and ten exact raw-f zeros. It alone occupies the strict near-linear validation bin. Its SSE is 2.898002867 (3.0763% of eta0 SSE) and 3.596540897 (4.1238% of equal3 SSE).
- The adjacent low-q2 bin has only five molecules. Their combined SSE is 2.487683314 eta0 and 1.660377394 equal3; median molecule SSE is 0.460131213 and 0.093515037, with P90 0.984108031 and 0.842237772. Thus the concentration is not solely accounting for ID14562, although five observations remain a very small sample.
- All six together are 0.0897% of validation molecules and contribute 5.7170%/6.0275% of eta0/equal3 SSE. Their absolute combined SSE actually decreases from 5.385686181 to 5.256918291 with the ensemble. The higher ensemble SSE share reflects its smaller full-set denominator; it does not mean the ensemble worsens the six as a whole. ID14562 individually worsens while the other five improve.
- Training contains ten strict near-linear molecules, with 72 of their 100 f labels exactly zero, but no C8H2 composition match. No exact connectivity match is expected under the connectivity split and is not independent evidence of unusual chemistry. Near-linear examples exist; sparse shape/composition support is a plausible diagnostic context, not an identified cause.

The original report uses zero-based state indices 0..9: its index6 is physical S7 and index7 is S8. This convention should be explicit in any publication or comparison with the earlier S7/S8 analyses.

## Limits and decision

The low-q2 tail is associated with concentrated error in this validation sample. Six molecules, including one prespecified difficult case, cannot establish a general geometry effect, a representation singularity, a label error or a reliable subgroup performance estimate. Coordinate shape is correlated with composition, molecular size and the distribution of target strengths; those factors were not controlled. Frequent small positive errors and a few large negative errors can coexist, so overprediction fractions alone do not establish broad positive bias (the five-molecule eta0 mean signed error is slightly negative). Exact zeros are supplied labels; earlier raw-source checks support retaining the case.

Keep these fixed strata as descriptive diagnostics for the approved seed and scratch comparisons when their results are complete. Do not exclude or downweight any evaluation case, adjust the objective, change either active study, or infer a new architecture requirement from this audit. The current evidence supports checking whether future gains reduce this concentration while preserving full-set raw-f R2; it does not authorize another experiment or test access.
