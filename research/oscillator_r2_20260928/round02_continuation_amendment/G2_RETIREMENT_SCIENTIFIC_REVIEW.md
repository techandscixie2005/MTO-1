# G2 retirement: scientific and operational review

Coordinator approved retirement after the completed epoch209 audit. This is an administrative truncation, not natural early stopping or a completed training schedule.

## Evidence and correction

Read-only strategic snapshot: last completed epoch209, epoch210 cursor119744/120355; best validation objective0.12957936948799922 at epoch41, latest0.15418163048019318. The run had 168 completed epochs without a new best. Its best oracle-energy native-f validation R2 was approximately0.38078, below eta0 native-f0.405294 (these are different energy assumptions and must remain labelled).

The earlier estimate of a natural stop at epoch200 was wrong. ReduceLROnPlateau ran before the early-stop check and reduced LR at epochs92,143,194. The rule requires at least50 epochs since the last reduction, in addition to minimum200 epochs and150 bad epochs. With no subsequent best improvement, the earliest natural stop is epoch244, before a prospective fourth reduction at245. Roughly34 more epochs, about85 minutes at the audited timing, remained from near epoch210.

The long plateau and weak existing validation evidence justify releasing this clean GPU for a controlled objective comparison. This decision uses validation and training evidence only.

## Required operational treatment

Signal only the verified G2 process from its receipt (audited PID566514, GPU2) after checking current process identity and cwd. SIGTERM is batch-safe: trainer saves model, optimizer, scheduler, RNG, sample order and cursor before returning. It does not guarantee an epoch boundary; report the actual retained checkpoint epoch and cursor, including any partial epoch. Retain and hash best.pt and last.pt. Do not create FIT_COMPLETE or alter original run records to imply natural completion.

The original supervisor records STOPPED_REQUIRES_EXPLICIT_RESUME and eventually CAMPAIGN_BLOCKED rather than automatically restarting G2. Preserve and document that intentional status; do not disturb other workers. Its uniform post-campaign evaluation may be withheld.

The original loss-pilot queue accepts only natural FIT_COMPLETE. A separate additive, reviewed queue amendment must explicitly verify administrative retirement and worker exit, hold the shared GPU2 lock, and require clean/idle hardware. Stop and replace the original waiting queue only after proving it has no active pilot worker. Do not edit any source hashed by the running architecture experiment. Model, loss, data, seeds, selection and training budgets stay unchanged.

This document authorizes no independent stop; the coordinator approved the executor to perform it after the queue amendment review.
