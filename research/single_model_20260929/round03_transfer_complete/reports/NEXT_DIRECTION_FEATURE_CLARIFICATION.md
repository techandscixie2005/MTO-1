# Feature control for a possible post-round03 readout

This clarifies the analysis-only note `NEXT_DIRECTIONS_AFTER_ROUND03.md` (SHA `dd9bcf2130df698258fbe54c1aca12b5f4b0ddaa985738fe65387db22c9b291f`). It does not change round03 or authorize implementation/fitting.

The frozen affine maps use only native f. A nonlinear `g(E_hat, f_native)` adds predicted energy as well as nonlinear flexibility. Its gain over an f-only affine map cannot be attributed specifically to nonlinearity.

**Smallest identifiable follow-on, only if round03 passes its existing gate:** prepare the two matched shared **f-only** nonlinear heads, keeping the same held rows, target labels, prediction-source pair, deployment full baseline, loss, sample order, initialization, optimization budget and nonnegativity rule. Compare each with its frozen f-only affine map and retain the native/historical-calibration references. Avoid state-ID, energy/gap or other context additions in this first comparison. This holds the input information fixed while testing added function flexibility; optimization differences and source-quality confounding must still be reported.

If predicted E is scientifically necessary, first add a matched linear E/f comparator (including intercept) with identical feature scaling and application constraints. Then distinguish added-input benefit from nonlinear benefit. That larger comparison is not the smallest follow-on and has no current approval.

Any future head pair still compares prediction-source procedures, not a perfectly isolated membership mechanism: the temporary source has less training data and different quality. A gate pass only permits preparation, and root must review state/tail tradeoffs before authorizing a fit. A gate failure does not justify automatically expanding features or head capacity.
