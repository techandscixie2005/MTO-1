# Independent novelty and control review

**PASS for the analysis-only note** `NEXT_DIRECTIONS_AFTER_ROUND03.md`, SHA `dd9bcf2130df698258fbe54c1aca12b5f4b0ddaa985738fe65387db22c9b291f`. This is not approval for implementation or fitting. Root will interpret the fixed round03 results before choosing either preparation.

## Historical finding

The audited records contain no true matched fresh-initialization right-F/raw-M-decorrelation factorial. Round01 applies those factors to the validation-selected eta0 epoch33 model. Round02 freezes that model and trains F only. Earlier scratch studies changed readout, objective, width or seed; they are not a prior test of this joint-learning question. This is a statement about the reviewed evidence, not proof of absence throughout every repository revision.

The note correctly avoids using those older results as a clean architecture rejection:

- Seed23/37 selected native validation R² are .368708895/.394455334; later TRAIN fitting improves while validation squared error deteriorates. Their fixed100 budgets differ from seed11's236-epoch stopping history, so this is not a matched-budget seed-variance estimate or proof of a capacity ceiling.
- Chan64 G1 gives .40525459 under eta1; G3 gives .36810069 with p_outer and E²-weighted trace. Their factors are confounded. G2/G4 used unsupervised energy heads and oracle-energy f diagnostics, which are ineligible native predictors.
- The100-epoch scaled-objective scratch MTO/direct-f results .373492948/.343781976 do not test right-F or raw-M decorrelation.
- No controlled dropout/weight-decay sweep was documented in this audit. That limited finding is not a repository-wide absence claim.

## Minimal distinguishing control

For a future F-only question, a contemporaneous original-MTO scratch control must share base initialization tensors, TRAIN order, objective, optimizer, schedule, budget and validation-SSE selector with identity-F. Constructor RNG must not alter the common base or data order. The historical eta0 selected checkpoint is an incumbent comparison, not the causal control. A two-arm pair cannot support decorrelation or interaction conclusions; those require corresponding factorial arms.

The final note now distinguishes a common realized learning-rate schedule from an identical adaptive scheduler rule that may produce different schedules. The protocol must specify which contrast is intended before fitting. A source-training or selection gain in control alone cannot be attributed to F.

The other direction remains conditional on round03's existing gate and would compare identical observable heads trained from different prediction sources on the same rows. It does not claim hidden-coordinate alignment or a causal explanation of current null results. Both directions preserve single-model deployment and prohibit prediction averaging/test selection.

## Remaining limits

Active-adapter null results and error-distribution differences support questions, not a mechanism diagnosis or promised gain. The note correctly separates geometry-computable AO overlap/dipole integrals from unavailable MO/X/Y response information. AO conventions and an empirical structured readout remain unvalidated; the direct PSD head already spans the output tensor range. Root deferred that implementation.

## Evidence reviewed

- `research_state/history_baseline.md` and completed `ROUND01_DECISION.md` / `ROUND02_DECISION.md`.
- Server `research/oscillator_r2_20260928/chan64_campaign_completion/CAMPAIGN_REPORT.md`.
- Server `research/oscillator_r2_20260928/eta0_seed_replication/completion_receipts/SEED_FAMILY_SCIENTIFIC_CLOSEOUT.md`.
- Server `research/oscillator_r2_20260928/CONSOLIDATED_RESEARCH_CLOSEOUT_20260929_FINAL.md` as previously audited baseline index.
- `ao_integral_feasibility/VERDICT.md` and its independent CPU/metadata evidence.

This review read existing source reports only. It did not inspect new calibration outcomes, load models/target arrays, poll GPUs, modify the running source or authorize additional work.
