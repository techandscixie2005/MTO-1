# Independent review of the validation state diagnostic

**Pass.** The three frozen validation prediction files match the original raw `f`/`E` labels, masks, IDs and split indices exactly. The equal-three prediction is the mean of the three native-`f` arrays. The protocol precedes the multi-model result but follows a disclosed eta0 feasibility check.

I independently reproduced eta0's offset1 low-gap 2,673-pair SSE (7.3472594604), pair-sum ratio (0.6795052154), adjacent-error correlation (−0.3229513109), and local-swap fraction (0.3027078899). Both disjoint offsets partition their own adjacent pairs correctly. A separate 2,000-draw connectivity-group calculation reproduced the ensemble-minus-eta0 pooled ΔR² interval [0.0253173997, 0.0445021358, 0.0621523010] to floating-point precision.

The interpretation correctly describes mixed stratified evidence: 17 sufficiently populated cells, with stronger low-gap cancellation in only 10 for each eta0 and the ensemble. It does not infer a sorting fix, relabel accuracy, or treat truth-informed swaps as deployable. Results remain exploratory and conditional on selected checkpoints.
