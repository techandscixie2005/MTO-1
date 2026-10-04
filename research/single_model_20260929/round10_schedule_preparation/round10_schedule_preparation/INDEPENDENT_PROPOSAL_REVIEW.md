# Round10 independent proposal review

**PASS for root's preparation decision; no execution authority.** The exact proposal is a fixed learning-rate contrast on original PSD MTO: `.001` throughout 60 epochs versus `.001` for epochs 1–30 and `.0003` for 31–60. No blocking finding remains in the documents. Implementation, model construction, training fixtures and production still require their stated later decisions and reviews.

## Scientific question and historical limits

The source-backed historical ledger contains initial-LR variants, validation-driven plateau schedules and a direct-f scratch schedule. Their objectives, heads, initialization, splits or realized budgets differ. Within those audited records, there is no matched test of this exact fresh-v2 original-PSD/LE+Ls/WD0 schedule contrast. This is bounded evidence, not proof that no such run exists elsewhere.

The comparison tests the complete lower-late-LR recipe. Its epoch-rate sum is .039 versus .060 for control; equal updates do not equal cumulative LR. A gain cannot isolate the drop boundary, identify an optimal schedule or show superiority to a lower constant LR. Late validation decline does not establish excessive LR, overfitting or an optimizer failure. No extra arm or sweep is needed to answer the stated narrow question.

The contemporary control is necessary given the realized variation among prior nominally matched seed11 controls. Historical controls are descriptive references, not independent-seed confirmation or evidence of the cause of that variation. Both new arms must use the same source, diagnostics, initial tensors, TRAIN-only statistics, masks, ordered 135-tensor optimizer membership and permutations.

## Exact schedule and recovery contract

- Epoch0 is an eligible initialization evaluation with zero updates. Both training runs retain 60 epochs and 112,860 updates.
- Candidate updates 1–56,430 use .001; update 56,431 begins .0003. Assign the absolute table value before each epoch's first minibatch.
- Keep Adam/AMSGrad moments, per-parameter counters, clipping, default dispatch and `grad=None` behavior. WD is zero in both arms. No optimizer reconstruction, moment rescaling or metric-driven scheduler is allowed.
- Committed epoch30 stores its used LR .001. Restore complete model/Adam/RNG/order state, validate the saved LR, and assign .0003 for epoch31. An incomplete epoch31 replays from committed30; completed60 refuses relaunch.
- Future implementation must replace constant-LR assertions coherently across runtime, recovery, metadata inspection and terminal analysis, while retaining the roster and optimizer-option checks.

The proposed later six-update boundary fixture distinguishes synthetic schedule positions30/31 from actual Adam counters1/2. It does not claim to reproduce a 30-epoch optimizer state. Its source, exact toy-call budget and manual tolerances remain future pre-execution review items; no such numerical test was performed for this proposal.

## Selection, interpretation and allocation

Earliest minimum pooled native validation SSE over epochs0–60 remains the selector, with fixed60 separately reported. All 66,860 raw-f labels and zeros remain included. The numerical screen requires candidate R² at least .003 above both the contemporary selected control and retained Round05 control45, R² .44716940136585204. TEST remains sealed; v2 is a historically exposed repartition, not an external confirmation set.

The final clarification is necessary: a selected candidate from epochs0–30 has not experienced the LR change. Its gain alone cannot support a schedule-benefit claim or justify schedule-confirmation seeds. Best prefix and post-drop values may be described from existing histories; they do not replace the selector, add a threshold or authorize extra inference. Root retains the allocation decision, and even a numerical pass launches no work automatically.

Paired component intervals remain conditional on reused validation and selected checkpoints, excluding training-seed uncertainty. No interval is claimed against the aggregate-only retained reference. State/tail/energy tradeoffs and independent matched-seed confirmation remain required considerations before an accuracy claim.

## Evidence and scope

Manual review covered the final protocol/settings, bounded historical ledger, existing optimizer/recovery source and published aggregate evidence. A separate stdlib metadata check rehashed all 38 reference files, both final documents and all four preserved draft files, and checked the explicit LR tables, transition indices, roster/statistics/split pins and allocation metadata. It imported no scientific package and opened no model, checkpoint payload, target or prediction array. No numerical experiment, inference, fit or previous completed stage was repeated.

Exact bytes are bound by INDEPENDENT_PROPOSAL_REVIEW.json and INDEPENDENT_REVIEW_MANIFEST.json. The three reviewer-owned history records under `../post_round09_schedule_audit/` are included in that closure. Referenced private paths are provenance only, not an archival payload allowlist.
