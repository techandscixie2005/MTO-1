# Prespecified schedule-control advancement rule

Root clarified this rule before observing any pilot fit outcomes. The objective is the best verified single-model predictor.

If the matched control's selected native validation R² improves by at least0.003 over its epoch0 anchor under the new lr1e-5 schedule, that schedule is eligible for independent-seed confirmation even if no adapter or decorrelation arm beats it. This tests whether slow continuation itself helps.

An architecture candidate remains eligible when its selected native validation R² improves by at least0.003 over **both** the matched control and epoch0. These are exploratory advancement thresholds, not proof of improved generalization. Report per-state, bright-tail, energy and fixed-calibration comparisons and confirm with independent original-training seeds. Predictions must remain single-model/single-checkpoint.

This amendment changes no training, checkpoint selector, code, frozen manifest, data, labels or test-access rule. A positive control result must not be discarded merely because an architecture hypothesis failed.
