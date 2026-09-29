# Bounded calibration-history check

Scope: read only the known historical `research/oscillator_r2_20260928/calibration/REPORT.md`, `calibrate.py` and `frozen_choice.json`. No broad repository search, model inference, target-array inspection, fitting or test scoring occurred.

The scripted candidate family was exactly identity, nonnegative global scale, per-state scales blended25/50/75/100% toward the global scale, and a global affine map clipped at zero. The report also discloses an earlier exploratory unregularized per-state affine calculation. This known family contains no isotonic, spline, polynomial or flexible f-only neural map. That is a scoped finding, not a repository-wide absence claim.

The output clipping is a fixed nonnegativity transform. It does not fit a free nonlinear shape, and the selected positive slope/intercept make it inactive for nonnegative native predictions. The selected map was `.8511830211044088*f + .003725185373211049`, with OOF validation R² about.41665766 and full-validation-refit R² about.41811924.

Historical calibration used five molecule-ID folds of the existing outer validation predictions from the same full-TRAIN eta0 source. All ten states of a molecule stayed together. It then refit the selected family on all outer validation; the neural checkpoint had already depended on validation. Its test evidence was historically exposed. This did not test a fresh source trained without the internal TRAIN calibration targets.

Round03's distinct question is transfer from a label-clean internal-TRAIN source, compared with in-sample full-baseline predictions on the exact same rows, while both deployment maps act on the same full baseline. A possible f-only nonlinear follow-on would add controlled flexibility to that source-provenance pair only after the current gate passes. Novel terminology is insufficient rationale: do not repeat an unchanged validation calibration or historical nonlinear h-only residual merely by calling it a head. The earlier feature-control clarification remains necessary; E/f inputs require a linear E/f comparator if nonlinearity is the intended contrast.

Historical script SHA: `7ba2de112aae8deaebe3d157840509d61143b390b6a95cb705946943700ab231`. Frozen-choice receipt SHA: `dc1ab25eb87b05499fdb0dbe7d50dbcc6d7cd1a6e2ed16b5eeaf30de9616aee2`. Its validation report SHA: `be80a185ffa436ecd4770c3735d92d05cb877360b31e1aef433066fff2eb42a3`.

No new fit or preparation is authorized by this check. Interpret the fixed round03 result first, retain all state/tail tradeoffs and the strongest eligible calibrated comparator, and preserve the absence of fresh holdout confirmation.
