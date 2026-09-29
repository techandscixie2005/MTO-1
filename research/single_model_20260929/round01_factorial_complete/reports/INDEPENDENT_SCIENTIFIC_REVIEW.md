# Independent round01 review

**PASS for integrity and the null-result interpretation. No recipe is promoted.**

The CPU-only review script freshly checked all41 frozen source/data/checkpoint hashes, all four normal20-epoch completions, exactly0..20 history rows, native-SSE earliest-minimum selection, actual best/last hashes, resumed optimizer/RNG state and every seed11 TRAIN order. It independently decoded only selected validation label rows and recomputed pooled/per-state/bright-tail metrics for all66,860 labels. All selected model tensors equal the starting checkpoint and each other; numerical replay differences are not improvements. The administrative model-capacity interruption did not interrupt the fits.

The source, input and output receipts for summary, mechanism, gap, plot and false-bright diagnostics match. Selected-checkpoint false-bright bins, quantiles, error concentrations, cross-products and correlations were independently recomputed from the retained validation arrays. Epoch20 replay scores agree with original history within1e-7; their bins/state counts/SSE/cross-product algebra agree. Epoch20 arrays were intentionally not exported or retained by that diagnostic, so this review did not perform a second model inference. Mechanism checkpoint hashes and the fixed256 TRAIN subset are verified. Drift R² measures agreement with the starting model, not predictive accuracy.

## Scientific conclusions

- All four arms select epoch0. Native anchor R² is approximately.405294118 and fixed calibration approximately.41811924. No architecture or schedule meets the≥.003 promotion rule.
- At epoch20 control pooled SSE rises by6.1206, while energy MAE, raw-f MAE, true-bright tails and eight per-state SSEs improve. S7/S8 worsen by9.3017/1.5589. The result is not uniform degradation.
- The all-label, posthoc brightness partition attributes10.0074 additional SSE to true-dim/predicted-bright labels under the fixed TRAIN q99=.2412 threshold; other bins improve. The top0.1% errors grow from22.83% to31.95% of SSE. No label was removed, relabeled or rescored to improve a primary score.
- F moves its input about.75–.78%; decorrelation reduces overlap beyond control. These interventions were active but ineffective under this schedule. Near-zero same-epoch factorial interaction does not support synergy. These observations do not establish a shared physical cause.
- Global S7/S8 error anticorrelation disappears and summed-error SSE worsens. Local small-gap cancellation is descriptive, not evidence that state permutation or strength redistribution explains the whole decline.
- The original gap-bin audit uses selected epoch0 predictions only. Its gap associations also reflect state/brightness composition; it cannot diagnose the final-epoch deterioration or prove a degeneracy problem.

## Precision and scope caveats

The later direct-adjacent-pair diagnostic used the rounded cutoff.0287eV rather than the exact TRAIN quantile.028699999999999726. Five adjacent-pair assignments differ, including two S7/S8 pairs. Its519 S7/S8 small-gap pairs refer to the rounded cutoff. Preserve these outputs with this clarification; primary metrics and selection do not depend on this subgroup.

Online TRAIN losses are batch means at changing weights, not a full fixed-checkpoint TRAIN raw-f evaluation. Historical-test evidence is already exposed; the new round is validation-only, and no fresh holdout exists. Legacy evaluation loads containers that include test arrays but does not index, infer on or score test examples. No physical wavefunction phase, NTO or transition-density reconstruction is demonstrated.

## Smallest next question

The root's paired frozen-base F comparison is justified as a bounded change in trainable parameters and readout location. Compare the same adapter under LE+Ls and LE plus variance-normalized raw-f MSE, fixed coefficient1 and the existing gentle schedule. Historical frozen h-only residual already used raw-f MSE; the distinction is the new pre-CG location with coupled E/A effects through a frozen decoder. It is not evidence that freezing will prevent rare outliers. Raw-M decorrelation is constant and must be omitted. Complete full-TRAIN variance provenance, real cache/gradient/update parity, strict parameter/buffer freeze checks, source review and archive-before-launch gates before any fit.

Detailed evidence: `INDEPENDENT_TERMINAL_INTEGRITY.json`; root decision: `../current_state/ROUND01_DECISION.md`. Review outcome authorizes closing round01, not launching round02.
