# Final20 replay interpretation and bounded case audit

Date: 2026-09-29. Stored-validation-array analysis only; no new inference, training, test-label use or evaluation change. Checked FINAL20_REPLAY_RESULTS.json SHA256ce60e54cf0a8165f73309d54d9dc28d010a220b4fd08149f629ab03ca1dc4d12 and its recorded runtime NPZ hash. Independently recomputed all three global SSEs, exact signed/base/delta identity, all eight strata and top-k deterioration aggregates; all matched. Final R2=.3049378477 reproduces the recorded run within numerical tolerance.

## Main interpretation

Against eta0, total SSE increases15.897059. The jointly changed base increases SSE20.010963; adding the residual and abs reduces it4.113904. The residual improves this final checkpoint relative to its own changed base, yet cannot recover eta0. This is an inference decomposition, not the counterfactual result of training without a residual.

The S7/S8, below-q90, nonnegative-signed cell contributes13.542595 extra SSE,85.19% of the net increase. Its206 negative-signed counterparts improve SSE by.213602. Thus the dominant added errors are positive overshoots, not negative-sign folding. Across the full validation set, mean signed prediction error is -.000324 for eta0, -.002061 for the changed base, and -.000461 after correction: a global positive-bias description would be misleading.

One molecule, ID14562, accounts for11.600012 extra SSE (72.97% of net deterioration). The ten most deteriorated account for15.463269 (97.27%). These are shares of NET increase: improvements elsewhere offset deterioration, so shares are not probabilities and larger groups can exceed100%. This strongly supports concentrated large errors rather than uniformly worse average error. The case remains in every reported metric; no clipping, exclusion, state filtering or posthoc replacement is proposed.

## ID14562: ten-state audit

Validation row771; dataset index14536. All ten raw printed f targets are zero, with E/A/f validity masks true. Energies below are eV. Base is the changed final20 tensor-derived intensity; delta is the additive correction before abs. Final is the emitted nonnegative value.

| State | True E | Eta0 E | Final E | True f | Eta0 f | Final base f | Delta f | Final f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|3.1588|3.505665|3.297156|0|.000019|.000765|-.011516|.010751|
|2|3.2447|3.931377|3.651466|0|.008626|.019801|-.011002|.008799|
|3|3.2447|3.764996|3.661298|0|.002803|.011250|-.011232|.000018|
|4|4.8656|4.867785|4.941291|0|.008487|.044198|-.018880|.025318|
|5|5.0078|5.150657|5.245938|0|.064989|.040361|-.019658|.020704|
|6|5.0078|5.103636|5.312858|0|.005567|.022253|-.018403|.003850|
|7|5.8582|6.101417|6.400966|0|1.665143|3.603086|+.001774|3.604860|
|8|5.9773|6.236992|6.231224|0|.344805|1.242946|-.019422|1.223524|
|9|5.9930|6.341969|6.403235|0|.037106|.049713|-.045234|.004479|
|10|5.9930|6.327747|6.153734|0|.025029|.098248|-.029777|.068471|

Across these ten states, total f is truth0, eta02.162575, changed base5.132622 and final4.970774. Molecule SSE is2.898003,14.543868 and14.498015 for eta0/base/final. The already-large eta0 error is amplified by continuation. Within this ten-state window, relabeling or permuting states cannot repair the f error: all targets are zero, so the sum of squared predicted intensities is permutation invariant. This is principally excessive predicted amplitude in this case. No statement is made about unobserved higher states or a full-spectrum oscillator-strength sum rule.

Adjacent energy gaps E(s+1)-E(s), states1->2 through9->10:

- Truth: [.0859,0,1.6209,.1422,0,.8504,.1191,.0157,0].
- Eta0: [.425712,-.166381,1.102789,.282872,-.047021,.997781,.135575,.104978,-.014223].
- Final: [.354310,.009832,1.279994,.304646,.066920,1.088108,-.169743,.172012,-.249501].

There are repeated true energy levels and predicted ordering violations; final S7 lies above S8 and S9 above S10. These may matter for broader state-allocation research, but cannot explain away this molecule's all-zero-f mismatch. S7 predicted E grows only4.91% from eta0 while its base f grows116.38%; the implied tensor trace grows106.26%. S8 E changes-.0925%, while its base f grows260.48% and implied trace260.81%. Thus energy multiplication alone is not the principal source of the amplitude growth. These ratios are algebraic decompositions, not causal parameter attribution.

## Bounded input/provenance checks

IDs/indices/raw validation truth align exactly between replay and frozen eta0 export. The identity record is [14562, '[H][C][C][C][C][C][C][C][C][H]', true, 'validated_covFactor=1.2']; its conservative connectivity key occurs once in the identity inventory. It is bond-order/stereo agnostic and must not be used as an authoritative bond-order string.

The input has8 carbon and2 hydrogen atoms. Coordinates are finite; minimum interatomic separation1.06252 Angstrom and maximum11.02958 Angstrom. Centered-coordinate singular values [11.382488,.014336,.000170] describe a nearly linear geometry. All52 cached directed edges exactly match pairwise radius5 Angstrom neighbors; no duplicate/self/out-of-range edges were found. These limited checks show no obvious coordinate, indexing, cached-edge or mask anomaly. No raw source-file reparse, electronic-structure validation, nearest-neighbor search or chemistry literature search was performed; therefore label correctness is not independently proven beyond the existing provenance and masks. Printed zeros are valid measured targets at the stored precision, not a claim of exact mathematical zero intensity.

## Consequence for the authorized frozen-head diagnostic

The replay strengthens the rationale for freezing eta0 h/E/base: it prevents the newly amplified base spikes while allowing the same correction head to learn. The current residual already helps its own final base globally, but this does not guarantee that a head trained on the frozen base will improve validation. Eta0 also has this hard case, so freezing preserves that initial error unless the learned correction reduces it. Keep the accepted protocol, full validation metric and all molecules unchanged. Do not tailor a loss, clipping threshold or special-case branch to ID14562. No architecture promotion or additional training experiment follows from this case audit.