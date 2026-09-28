# G2 retirement amendment: PASS

Reviewed the additive g2_retire.py and pilot_queue_admin.py. Both current source hashes and the unchanged legacy training/queue/config hashes are sealed in G2_RETIREMENT_AMENDMENT_REVIEW.json. Independently verified all44 architecture hashes and all35 original pilot preflight hashes remain unchanged. Syntax, exact-process negative cases, and missing-marker fail-closed checks pass.

The executor may stop only the verified inactive waiting queue and G2 process, retaining actual best/last checkpoints and full resume state. A signal during validation can finish an epoch; both valid stopping statuses are checked against the retained checkpoint. No FIT_COMPLETE is fabricated. The replacement queue requires matching administrative markers bound to this review, process exit, retained checkpoint hashes/state, and clean idle GPU2 under the shared lock. Downstream A/B/C training behavior, scientific settings, selection and budget are unchanged.

The corrected earliest natural stop is epoch244, because the third LR reduction at194 resets the required50-epoch wait. The original stopped/blocked supervisor status is intentional and must remain visible. Actual execution and stopping epoch/cursor are recorded separately by the executor.
