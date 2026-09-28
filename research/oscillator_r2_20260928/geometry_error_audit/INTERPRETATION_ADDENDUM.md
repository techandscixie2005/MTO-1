# Geometry audit interpretation addendum

This addendum clarifies labels and concentration accounting without changing the audited metric JSON or report. The independent scientific review is `INDEPENDENT_SCIENTIFIC_REVIEW.md`.

- The report's state column is a zero-based matrix slot: slot 0 maps to physical state S1, through slot 9 mapping to S10. Thus slot 6 refers to S7.
- For the six lowest-q2 validation molecules (the single strict near-linear case plus five in the next low-q2 bin), total SSE is 5.385686 for eta0 and 5.256921 for equal3. Equal3 reduces absolute SSE by about 2.4%, while its share of its own full-validation SSE is higher (6.028% vs 5.717%) because equal3 reduces total validation SSE more strongly.
- The strict q2 <= 1e-5 validation stratum contains only molecule 14562. The five molecules in the adjacent low-q2 stratum remain descriptive evidence of concentration in lower-q2 geometry, not a causal effect or a basis for sample removal.
- The reviewer independently recomputed the diagnostic and found no causal or label-error basis. No model, benchmark, or sample inclusion decision changes.
- Root authorized reuse of the frozen training-derived shape bins for saved-prediction diagnostics after seed/scratch runs complete. This authorizes analysis only of existing saved predictions; no trainer edits or new inference.