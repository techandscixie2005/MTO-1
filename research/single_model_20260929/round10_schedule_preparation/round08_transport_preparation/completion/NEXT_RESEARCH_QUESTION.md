# Next question: one regularization contrast on the original model

Discussion only. This note authorizes no implementation, numerical check, fit, seed allocation or TEST access. Round08's frozen gate and outcome remain unchanged. Root decides any later proposal after closeout publication.

## Recommendation

Prefer one fresh original-PSD comparison of **Adam AMSGrad with coupled L2 weight decay 1e-4 versus weight decay 0**, with the original LE+Ls objective, common initialization and TRAIN order, fixed LR .001, batch64, clip5, and 60 epochs. Apply the single coefficient to all original trainable parameters through the optimizer; keep the frozen dormant schemas excluded as before. This is a proposed fixed setting to review before preparation, not a value validated by this campaign. No coefficient sweep, dropout, schedule change, transport, changed readout or raw-f objective would enter this contrast.

The question is falsifiable: does this one regularized recipe improve selected pooled native validation raw-f R² by at least .003 over BOTH its contemporary zero-decay control and retained .44716940136585204, while documenting every state, bright-tail and energy tradeoff? Fixed60 and complete trajectories remain secondary; the existing earliest validation-SSE selector remains common. A passed allocation screen would still require a later decision and within-seed independent confirmations, not prediction averaging.

Coupled L2 is deliberately specified rather than called AdamW: in Adam it enters the gradient and moment estimates, so the contrast tests that complete optimization recipe. With the inherited clip_grad_norm_ before optimizer.step ordering, Adam adds its weight-decay term inside step AFTER clipping; clip5 therefore does not bound the total effective gradient in the same way. This proposed ordering and the exact parameter roster must be frozen before execution, without validation-driven adjustment. The coefficient has no demonstrated optimum or guaranteed benefit; any result would concern the complete recipe, not an isolated generalization mechanism.

## Evidence and limits

The bounded historical ledger found weight decay zero in all 23 inspected early configurations and no clean backbone/dropout/weight-decay ablation. Historical channel and LR trials used different objectives or adaptive budgets; they do not validate a present coefficient. See `../../post_round07_bottleneck_audit/HISTORICAL_CAPACITY_AUDIT.md` and its exact evidence manifest. This is novelty within the audited records, not a repository-wide absence claim.

Recent fixed-budget runs repeatedly reduce the logged TRAIN loss while validation performance deteriorates after the selected epoch. TRAIN values are pre-update trajectory averages, not fixed-checkpoint evaluations; these observations motivate a generalization question without proving its cause. Repeated original controls differ substantially despite matching declared initialization/order/settings, and executable graphs and workloads differ. Beating only a weak contemporary rerun is insufficient.

Round08's local and neighbor residuals become active during training. Their paired validation gains therefore do not justify tuning transport scale after a failed retained-reference gate. A capacity change would introduce a different structural question and higher compute before a basic regularization contrast has been tested. This recommendation neither establishes a representation ceiling nor attributes errors to labels, QC phase, or a missing physical mechanism.

The proposed pair costs about two current 60-epoch runs, roughly six GPU-hours based on recent measured training/validation durations, plus separately authorized bounded preparation. Memory should be similar to the original model; that estimate is not fresh resource admission. Existing TEST remains sealed and cannot be used to select the setting. The target R² .60 is still unmet.
