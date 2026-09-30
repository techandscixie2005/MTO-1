# Next controlled question after Round05

This is a decision note only. It authorizes no implementation, fit, new target access or hyperparameter search. Close and archive Round05 first.

## Recommendation: test the raw-f objective with the original PSD head unchanged

The smallest distinct next question is a fresh matched two-arm experiment: original PSD MTO under LE+Ls versus the same original PSD MTO under LE+Lf, where Lf is all-valid raw printed-f MSE divided by the fixed new TRAIN population variance `.0025089892829484074`. E and A remain coupled through native f=(2/3)E_Hartree tr(A); retain LE with coefficient1. Do not add F, decorrelation, a direct-f output head, calibration, dropout, weight decay, tail reweighting or a schedule change. A contemporaneous LE+Ls run with identical fresh base/order is essential. Preserve the existing split, TRAIN statistics, all labels, validation checkpoint policy and sealed TEST.

If root chooses this question, a bounded60-epoch matched seed11 pair using the existing fixed LR.001/AMSGrad/batch64 recipe would isolate the objective change within that recipe. Epoch0 remains eligible. Predeclare a +.003 validation R² candidate-over-control gate and full state/tail review before considering separate paired seed23/37 confirmation. The original control remains a reference; neither historical weights nor the preflight state may initialize either run. Estimated cost from Round05 is about2.7h concurrent or5.4 GPU-hours before engineering/headroom. These are proposed settings, not execution permission.

### Why this is new within the audited history

- Historical R².373493 was a different direct scalar-f readout/LE+Lf study with different energy initialization and a100-epoch schedule. It did not retain the original PSD head.
- Earlier original-PSD raw-f and E²-weighted trace studies fine-tuned the already selected eta0 epoch33 model at LR1e-4. They all selected their initial checkpoint.
- Round02 froze the selected backbone and trained only F; its raw-f arm made a very small validation gain.
- Original eta0 and seed23/37 used LE+Ls. The eta1/channel studies added Q supervision, not this raw-f objective.

Thus the audited record set contains no matched fresh-initialization original-PSD LE+Lf versus LE+Ls comparison. This is narrower than a repository-wide claim of absence. Prior raw-f failures remain negative evidence; changing initialization does not guarantee benefit. Exact history references are `HISTORICAL_NONREPETITION.md`, `history_baseline.md` and the historical scratch_readout_comparison config/protocol.

### What Round05 does and does not imply

The common control peaks at epoch45 and all ablations peak earlier, then worsen by60. After their selected epochs, TRAIN trajectory raw-f/energy/trace losses continue decreasing, while validation trace/raw-f error worsens and validation energy error slightly improves. F is active (mean relative change .54/.57 at60); raw-M decorrelation is measurably reduced. These are not dead-gradient runs. Tail tradeoffs differ: F improves selected true-bright RMSE but raises false-bright SSE; decorrelation reduces selected false-bright SSE while missing true-bright strength more severely.

This supports measuring the objective's effect directly before adding another learned readout. It does not prove that objective mismatch causes the failures, that the model has a capacity ceiling, or that raw-f training will fix rare large errors. Squared raw-f loss itself can be dominated by rare outliers. Gradient magnitudes and clip rates should be audited and reported without tuning a coefficient on validation. A failed pair should close that fixed recipe rather than trigger a broad loss/LR sweep.

## Deferred alternative: common PSD congruence with a scalar control

The reviewed QC-inspired proposal adds the same177-parameter invariant gate to tensor and isotropic arms. With R=sum A, S=tr(R), B=(R+epsilon I/3)/(S+epsilon), q=sqrt(tr(B²)/3), b=.25 tanh(g), compare A'_k=(I+bB)A_k(I+bB)ᵀ against A'_k=(1+bq)²A_k and an unchanged original reference. Both modified arms have identical inputs/gates, identity initialization and Frobenius perturbation norm. It introduces shared directional context from all states, not missing PSD expressivity or a demonstrated TDDFT mechanism.

The construction avoids transition-vector phase and square-root gauges, preserves PSD/O(3), and has a correct exact-degenerate block-sum property in its synthetic amplitude fixture. However, original LE+Ls does not identify physical orientations of A; the directions are latent features. NumPy algebra is the only evidence so far. Real FP32 autograd, near-zero conditioning, geometry symmetry and checkpoint-access preflight remain necessary, followed by at least a matched tensor/scalar/original comparison (roughly8h GPU-time at the existing budget). It would change the readout while leaving the training objective's raw-f alignment unresolved.

Prefer the simpler objective question first. Retain congruence as a separate empirical structural hypothesis if root later chooses it; do not combine it with the objective change in the first pilot. Ordinary anonymous polar-vector heads and naive AO transition densities remain invalid substitutes without resolving electronic-character/point-group issues. No new quantum-chemical labels or unavailable inference input should be assumed.

## Limits and next authority

The new partition is an audited internal repartition of historically exposed QM9S data. TEST stays sealed and current evidence is single-seed validation only. Target R²0.60 remains far above the current v2 reference .44717; neither proposed change is a forecast of reaching it. Root must choose a concrete next preparation after reviewed Round05 records have been downloaded to D:\MTO\archives and published. Source/preflight review, a new archive/publication gate and explicit execution authority remain required before any fit.
