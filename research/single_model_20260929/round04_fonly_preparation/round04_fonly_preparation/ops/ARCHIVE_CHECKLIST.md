# Round04 preparation archive checklist

Status: prospective only. No source freeze, archive bundle, staged publication or execution authority is created by this checklist. Namespace: `round04_fonly_preparation`. Monitor owns this `ops/` directory; science owns source and preflight; history owns independent review; root owns the preparation closeout and any later execution decision.

## Current authority

Read `current_state/ROUND04_PREPARATION_DECISION.md` and current coordinator. Independent protocol review must pass before the authorized implementation/preparation. The accepted function has four shared f-only basis coefficients, fixed TRAIN knots0.0549/0.2412 and common scale0.050109692115345626. No E, state index, context or hidden features are introduced. FP64 SVD and its fixed rank/condition failure rules are proposed fitting settings, not permission to solve real labels during preparation.

Allowed preparation is synthetic CPU testing, read-only contract/rank/support inspection of the frozen internal calibration cache, and CPU synthetic-geometry one-checkpoint/access checks with hand-set test coefficients. CUDA visibility must be empty before Python imports. Synthetic constants and diagnostics must be labeled as synthetic, with no experimental accuracy claim.

No real-data solve, new source prediction generation, validation-value inspection, test access, GPU allocation, production launch or validation reopening is authorized. A future execution must require a separate explicit decision bound to final source, review and verified publication; preparation cannot create that authorization. Publication itself does not authorize fitting.

## Gates before freezing an allowlist

1. Preserve the original documents-only three-file snapshot and its manifest before adding implementation; obtain exact preserved paths and hashes from science.
2. Freeze final protocol/settings, executable dependency closure, exact environment and reproducible CPU-only preparation commands. Preserve both original proposal and final protocol/review history.
3. Require meaningful synthetic solver/map tests: recovery, nested affine case, continuity, extrapolation, clamp/parity and all explicit fail-closed cases. Keep real calibration contract/rank/support aggregates separate from synthetic results; confirm no real coefficients were solved and no outer-validation values were accessed.
4. Require CPU one-checkpoint/access tests using synthetic geometries and hand-set coefficients, unchanged original full baseline, and no cache/source/QC-label dependency at inference. Post-fit real-geometry replay belongs to future execution, not this gate.
5. Require independent source/preflight review bound to exact final closure and root preparation closeout. Root's current decision grants preparation, not a real-data fit.
6. Freeze every reviewed source/log/aggregate/report/decision byte before packaging. Do not snapshot a moving handoff or silently omit failed initial checks. There is no finalization command or completed bundle yet.

## Prospective lightweight records

- Original documents-only snapshot, `PROPOSED_PROTOCOL.md`, `RATIONALE.md`, proposal manifest, independent protocol review and subsequent frozen protocol/settings.
- Final source files, dependency path/SHA mapping, source manifest, environment package versions, exact CPU commands, synthetic and read-only contract checks, logs, aggregate receipts, failure records and review/report manifests. Science supplies exact filenames after implementation; no broad recursive inclusion is permitted.
- Current preparation decision and closeout. Operational coordinator/science/history/monitor handoffs are reviewed at finalization. If a current historical handoff contains fitted coefficient values, preserve its path/SHA reference rather than recopying those values into this preparation bundle.
- `monitoring/CHECK_20260929T234954Z.json` and `monitoring/CLOSED_CAMPAIGN_20260929T234953.json`: the scheduled check found seven terminal owned runs, no incidents or pending Round03 continuation/archive.
- `monitoring/ROUND03_PUBLICATION_WORDING_CORRECTION_20260929T2350.json`: exact authorized sentence correction in current runtime/science handoffs, with before/after hashes; all frozen/archived snapshots preserved.
- `ops/HEARTBEAT_CLOSED_CHECK_20260929T2350.json` and `ops/record_completed_campaign_check.py`: one unchanged ACTIVE four-hour heartbeat and one existing four-hour cron, without duplicate schedules.
- `ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json`: completed parent receipt at4f9ae506. Copy other necessary derived publication receipts from D to scoped server records before including them in the new server bundle.
- This checklist, prospective inventory, eventual finalizer/explicit allowlist, final archive/content review and publication provenance.

## Exclusions and required order

Never package checkpoints, model/head tensors, fitted coefficient values or arrays, optimizer states, raw data/labels, identity/split/index arrays, prediction arrays, caches, credentials or raw QC logs. Do not copy excluded artifacts merely because a source manifest references them; record server path, hash and approved metadata only. Fixed protocol knots/scaling and explicitly identified synthetic hand-set test constants are permitted.

After all gates pass: inspect the explicit lightweight filenames, text contents and sizes; package server records; FIRST download to the new `D:/MTO/archives/single_model_20260929/round04_fonly_preparation` directory; verify every member hash/size; THEN stage byte-exact blobs with `--no-filters`, inspect all additions and diff, commit and non-force push, and verify the remote. Parent must be the current verified research-branch descendant of4f9ae50647f957ac7d68ab1a20200a79fa527cf9. Never rewrite earlier archives or the unrelated publication worktree.

The four-hour monitor stays unchanged. No GPU resource is allocated by this checklist. Honest failed preparation records may be archived after root closeout; do not fabricate success or omit failures.
