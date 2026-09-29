# Round03 completed-round archive checklist

Status: prospective operational checklist only. No completed-round bundle has been created from this checklist, and it does not authorize new model fitting. Preserve the immutable round03_transfer_preparation archive and its published commit 1b34e839335177e1fd582944549be4f181dee3c0.

## Ownership and gates

Science owns the active source run and the already-reviewed affine continuation in ops/ROUND03_RUNTIME_HANDOFF.md. The monitor checks only registered jobs on the persistent four-hour schedule and handles operational receipts/archive preparation. History owns independent scientific/terminal review. Root owns the next decision and new-fit authorization.

A successful round is ready to archive only after source epoch33 completion, verified process exit and source checkpoint hashes, frozen affine coefficients and exports, one completed fixed validation comparison, final aggregate analysis, independent review and root decision. Source FIT_COMPLETE by itself is not full-round completion. Preserve coefficients and completed validation; never refit or rerun for a preferred score. Honest failed/aborted rounds follow ops/FAILED_ROUND_ARCHIVE_POLICY.md after the coordinator records that decision.

## Supplemental records required

1. AO-integral feasibility package: ao_integral_feasibility/MANIFEST.json, SHA256 82e09937178c5474db09cd5b2890189fb4d4e857f887cc58262a474ac8059172. Verify its exact six listed files at finalization:
   - synthetic_audit.py
   - SYNTHETIC_RESULTS.json
   - METADATA_AUDIT.json
   - audit_frame_metadata.py
   - FRAME_PROVENANCE.json
   - VERDICT.md
   This is bounded CPU/metadata feasibility with cautious vector-frame provenance; no accuracy claim or new fit follows from it.
2. Startup/monitor evidence: ops/check_round03_startup.py, ops/ROUND03_STARTUP_CHECK.json, monitoring/CHECK_20260929T185538Z.json, ops/check_round03_continuation.py and monitoring/ROUND03_HEARTBEAT_20260929T194927.json, plus any actual later scheduled records relevant to the run. The scheduled heartbeat found the owned source healthy at epoch28; this is a progress observation, not a completion receipt. No extra polling solely to refresh an archive.
3. Heartbeat provenance: ops/HEARTBEAT_BEFORE_ROUND03.toml, ops/HEARTBEAT_AFTER_ROUND03.toml, ops/ROUND03_HEARTBEAT_VERIFICATION.json. Inspect these exact records for secrets as usual. The schedule/target/notification policy were preserved, and the continuation prompt follows the reviewed runtime handoff.
4. Current operational state: current_state/COORDINATOR.md, current_state/monitor.md, the science/history handoffs, ops/ROUND03_RUNTIME_HANDOFF.md, registry and exact launch/registration/admission receipts. Snapshot current bytes only when finalizing. Latest root coordinator was mirrored after the bounded-analysis decision with SHA256 915a0ff493f426b4ea7e22444aad72f2af4636798da4d4958f56219f21f24207. It defers both proposed preparations until Round03 interpretation and preserves the already-authorized affine continuation.
5. Publication provenance: ops/ROUND03_PUBLICATION_RECEIPT.json and exact derived download/stage/content-review/commit/publication receipts from the local preparation archive. Mirror locally generated receipts into a scoped server records directory before the new server bundle is downloaded.
6. Final analysis-only direction note and independent review, all under reports/:
   - NEXT_DIRECTIONS_AFTER_ROUND03.md SHA256 dd9bcf2130df698258fbe54c1aca12b5f4b0ddaa985738fe65387db22c9b291f.
   - NEXT_DIRECTIONS_REVIEW.md SHA256 92e4feec711d8333fb36576867c46c1a8a342127cbe9e3d6613deb8fa2101c09.
   - NEXT_DIRECTIONS_REVIEW.json SHA256 75d9dd87fd7a4720ca904948507a42ea54e0698cb7363be1097273144c2642de.
   - NEXT_DIRECTIONS_MANIFEST.json SHA256 818131c5cc4c5439f3d54bab7feda6710df44b7b1875b0c211880fb79f488557; verifies the exact three files above.
   Review PASS concerns novelty, controls and scientific wording only. Neither this package nor the archive checklist authorizes nonlinear, scratch-F or AO preparation/fitting. Root will first complete and interpret Round03.
7. This checklist and the versioned completed-round finalizer/allowlist, plus final source/affine logs, settings, immutable source manifests, aggregate all-label/state/tail results, checkpoint metadata, independent reviews and root decision.
8. Separate terminal-review preparation: round03_transfer/TERMINAL_REVIEW_CHECKLIST.md SHA256 16d3d1b0859037474ee8a5ffb2c79e3e9804e1f79b1cfff5b1d64f5808980ee1. This checklist is outside the frozen execution closure and is not a terminal PASS.
9. Separate analysis clarification: reports/NEXT_DIRECTION_FEATURE_CLARIFICATION.md SHA256 711eebede513d57b024012a65f9f6ec1bc3b5c7be9071140fe686f3827c6601e. It records the f-only versus E/f feature confound and gives no fitting authority.
10. Bounded historical calibration note: reports/CALIBRATION_HISTORY_CHECK.md SHA256 0e3c25ae3080ca547de9330075006b67836a76fd40d2fd81b3d3361f441e3034. The reviewed historical calibration family included identity, global scale, per-state shrinkage and clipped affine; the absence of flexible f-only nonlinear maps is limited to that reviewed family. No fitting or new data was introduced.

The prospective inventory builder ops/prepare_round03_completion.py only enumerates expected lightweight filenames and checks supplemental hashes. It does not query experiment processes, package a bundle, emit a completion allowlist, or claim readiness. Final source/affine analysis filenames and bound terminal review/root decision must be added after those records exist. Use a finalized allowlist only after all gates above pass.

Terminal source follow-up: include monitoring/ROUND03_HEARTBEAT_20260929T195916.json, which verifies all33 epochs, source exit and exact final/last hashes. Also include ops/ROUND03_TERMINAL_RESOURCE_OBSERVATION.json and science's eventual placement explanation: a short-lived MTO PID1186535/504MiB was observed on GPU0 and disappeared before a read-only ownership check. Preserve observations and distinguish incidental initialization from model inference using evidence; allocation size alone is not a scientific failure or evidence of harm.

## Required archive order

Freeze an explicit lightweight allowlist, inspect text contents/sizes and each supplement's hashes, package server records, FIRST download and verify to a new directory under D:\MTO\archives\single_model_20260929, THEN stage exact bytes with --no-filters, inspect every new blob/diff, commit and non-force push, and verify the remote commit. Use the current research-branch descendant of 1b34e839 as parent; preserve all earlier archives and the unrelated publication worktree.

Never download or publish model/checkpoint/optimizer binaries, raw datasets or labels, identity/index/split membership arrays, calibration/validation prediction arrays, caches, credentials or raw QC logs. Checkpoint path/hash/size metadata is allowed. No partial bundle may be labelled a completed round.

