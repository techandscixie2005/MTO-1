# Independent TRAIN reader review

PASS for the new TRAIN-only statistics stage. This does not authorize model fitting, validation scoring, or TEST access.

The reader checks the frozen split and independent verification, private index hashes, allowed partition membership, source archive hashes and member allowlists. It rejects TEST and unapproved VALID construction before file access. Only selected contiguous row blocks reach numeric conversion; compressed ZIP transport can pass other bytes without interpreting their values.

The statistics use all 120,355 new TRAIN molecules, original per-state-centered energy variance, mean squared Frobenius tensor norm, median atom count, pooled raw-f population variance and TRAIN-only quantiles. The formulas match the earlier audited FP64 definitions. Invalid or nonfinite labels cause failure rather than row exclusion. No real statistics were computed by this review.

Source inspection covered all four scientific files plus the owned CPU wrapper. Existing synthetic tests verify selected-row conversion and request rejection. Separate reviewer tests verified split-pin and verification-binding rejection, numeric strides for IDs/E/A, and scalar-reference normalization reductions. All passed without opening real corpus values or a model.

The CPU wrapper registers before spawning the child, preserves a single immutable attempt and records child identity, logs and terminal output hashes. An interrupted partial statistics record needs explicit inspection; it is not automatically rerun.

For the future runner, validation_authorized is an internal permission flag, not execution authorization. Its caller must pass the independently reviewed production gate before constructing a validation reader. The present receipt permits only the stated TRAIN statistics preparation.
