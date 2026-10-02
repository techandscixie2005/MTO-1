# Failed or aborted round archival

The successful-round finalizer requires all four fixed20epoch completion markers before it permits a **completed results** claim. It must never prevent preservation/publication of a genuinely failed or aborted round.

If the coordinator explicitly declares a round failed or aborted, create a separate failure archive with a distinct round ID. Include the declaration and scientific/operational decision, original protocol/source hashes, launch/identity records, all actual status files and logs, existing FAILED/FIT_COMPLETE/interruption receipts, checkpoint **metadata only**, health records, diagnostics, and attempted recovery records. Include every assigned arm, marking its actual state: complete, failed, intentionally stopped, or still running. Preserve absent/missing artifacts as facts. Never synthesize FIT_COMPLETE or invent completed epochs.

Use the same text-file content/size/credential checks and explicit allowlist, `package_records.py`, and `download_records.py`. Download and verify on D: first; then stage exact bytes with `--no-filters`, inspect all additions, commit and non-force push as a descendant of the current remote research branch. The failure manifest must say `failed_or_aborted_round`, identify the coordinator declaration, and list missing/unfinished records. A separate active incident archive can preserve failures while another arm is still legitimately running; it must not claim the whole campaign is terminal.

The declaration is a record requirement, not permission to stop or restart jobs. Interventions require exact owned-process identity and resumable-state checks plus coordinator direction; never touch unrelated jobs. Model-capacity errors or lost observer sessions are administrative events and are not evidence of training failure. Recheck actual /proc identity and terminal/failure receipts before classifying a job.

This policy changes no running fit, frozen source, checkpoint selection, validation label, schedule, or successful-round gate.
