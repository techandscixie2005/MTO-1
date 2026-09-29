# Additive bright-tail threshold provenance correction

Recorded at 2026-09-29T08:16:55.494342+08:00. No historical sources, numerical slices, candidates or selections are changed.

**The historical raw-f cutoffs q90=0.0546 and q99=0.2377 are validation-derived.** The authoritative original `ensemble_diagnostic/FROZEN_CHOICE.json` explicitly records them under `bright_thresholds_raw_f_from_validation`, and its historical evaluation rule calls them validation-defined. Later studies reused these fixed numerical cutoffs. They must not be relabeled TRAIN-derived merely because TRAIN metrics were also evaluated at those cutoffs.

**For the prospective final historical-test protocol only**, the actual TRAIN-only pooled raw-f quantiles are q90=0.0549 and q99=0.2412. They were recomputed before test access using all 120355×10 raw FP64 TRAIN labels, valid masks, original indices and NumPy `quantile(..., method="linear")`. Ties enter the bright side (`f_true >= threshold`), with the complement strictly below. No test values were loaded or used to set either threshold.

The existing Round09 geometry q2 boundaries were already fitted from TRAIN coordinates plus the prespecified near-linear cutoff; they remain unchanged. Tail-label thresholds and geometry boundaries have different provenance and should be named separately.

Affected claims are any earlier conversational or summary shorthand describing the joint tail/geometry set as entirely TRAIN-derived. The bounded audit inspected the original choice/diagnostic code and relevant descriptor protocol/review/closeout, scratch protocol/closeout, seed closeout, fixed-seven amendments/closeout, geometry protocol and final selection policy (exact files/hashes in companion JSON). Their historical tail values are generally described as fixed; the explicit TRAIN wording in geometry clauses applies to q2. No historical document in this audited list requires numerical recomputation or replacement. This additive correction governs downstream summaries: say “historical validation-derived f cutoffs; TRAIN-derived q2 bins.”

Root accepted prospective TRAIN tail cutoffs for the final protocol. This diagnostic provenance correction does not affect pooled raw-f SSE/R², F5 candidate selection, checkpoint hashes, the frozen mixture roster or any prior result. It authorizes no test read, model inference, training or extra analysis. The final candidate freeze must bind this erratum and the exact TRAIN quantile evidence.
