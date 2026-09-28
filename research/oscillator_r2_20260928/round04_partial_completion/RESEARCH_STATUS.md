# MTO-1 oscillator-strength research status

Updated 2026-09-28. This file is a handoff index for the ongoing research campaign. It does not replace sealed protocols, reviews, or per-round result records.

## Objective and decision rules

The primary objective is better pooled **raw native oscillator-strength R²**. The research coordinator remains responsible for choosing experiments and integrating evidence. Tune and select with validation data only. Historical test results below are exploratory because the split has been examined in earlier campaigns; use a newly sealed molecule-disjoint holdout before making a generalization claim. Do not edit any frozen protocol or choice record after it is sealed.

## Current benchmark and completed evidence

| Model or result | Validation pooled native-f R² | Historical-test pooled native-f R² | Status |
|---|---:|---:|---|
| Single MTO eta=0 baseline | 0.4052941183 | 0.45581511 | Checkpoint selected on validation; historical test has been exposed. |
| Equal three-model eta ensemble | 0.4494214633 | 0.48510176 | Exploratory historical-test gain +0.02928666; three forward passes. |
| Validation-fitted affine calibration | OOF 0.416658 | 0.459667 | Exploratory only; Δtest R² +0.003852 with molecule bootstrap 95% CI [-0.012204, +0.022829]. |

The calibration result is too uncertain to establish a reliable gain. The equal-weight ensemble improves the historical pooled test result and lowers test SSE 5.38%, but the test split is reused. Connectivity-group bootstrap gives ΔR² 95% CI [+0.003586, +0.056691] over 6,671 groups. The ensemble requires three models at inference. It slightly improves the validation-defined bright q90 subset, but test SSE worsens above the validation q99 threshold (672 transitions), and state slots 7 and 8 also regress. Preserve these limits in future summaries.

## Round index

### Round01 — post-hoc calibration diagnostic (completed)

A five-fold molecule-grouped validation screen selected a clipped global affine calibration. It raised pooled historical-test R² by 0.003852, but the paired bootstrap interval includes zero and bright-transition SSE worsens. Do not continue fitting calibrations to the exposed test split.

- Record: `calibration/REPORT.md`
- Protocol outputs: `calibration/validation_crossfit.json`, `calibration/frozen_choice.json`, `calibration/exploratory_test.json`
- Method: `calibration/calibrate.py`

### Round02 — matched native-f loss pilot (reviewed; queued at last archived receipt)

The reviewed screen compares three matched 20-epoch arms under a common native-f objective. After the additive amendment passed review, chan64 G2 PID 566514 was intentionally and gracefully stopped at epoch 215, cursor 114496/120355. Its best checkpoint remains epoch 41 (objective 0.12957937); last and best checkpoints remain resumable. The old queue PID 762340 exited without workers. Replacement queue PID 785736 launched unchanged-control worker PID 785960 on clean GPU2; its launch receipt records epoch-zero native-f validation R² 0.4052941077, passing baseline reproduction. Weighted and direct-f-matched arms follow. The original four-group campaign is expected to report `STOPPED_REQUIRES_EXPLICIT_RESUME` / `CAMPAIGN_BLOCKED`; this is intentional, no `FIT_COMPLETE` was forged, and remaining chan64 arms continue. Once they finish, report that campaign as partial and administratively stopped.

- Protocol and setup: `SCIENTIFIC_AUDIT_AND_PROTOCOL.md`, `pilot_config.json`, `pilot_objective.py`
- Trainer and queue: `train.py`, `pilot_queue.py`, `summarize.py`
- Review and preflight: `CODE_REVIEW.md`, `CODE_REVIEW.json`, `preflight.py`, `preflight_results.json`
- Retirement amendment and provenance: `g2_retire.py`, `pilot_queue_admin.py`, `G2_RETIREMENT_AMENDMENT_REVIEW.md`, `G2_RETIREMENT_AMENDMENT_REVIEW.json`, `G2_RETIREMENT.md`, `G2_ADMIN_STOPPED.json`, `G2_RETIREMENT_PREFLIGHT.json`
- Replacement launch receipt and log: `runs/control/queue_launch_receipt.json`, `logs/pilot_queue.log`

No result is recorded for this pilot yet. Its choice and analysis must use validation metrics; no test split is part of its plan.

### Round03 — equal three-model MTO ensemble (completed; exploratory)

The frozen ensemble averages native f predictions from `mto_eta0`, `mto_eta01`, and `mto_eta1` at exactly one-third each. Validation R² is 0.4494214633 versus 0.4052941183 for eta=0. On the reused historical test, R² is 0.48510176 versus 0.45581511, with paired ΔR² +0.02928666 and a connectivity-group bootstrap interval [+0.003586, +0.056691]. The improvement is promising but remains exploratory; preserve the three-model inference cost, upper-bright-tail loss, and state-slot 7/8 regressions.

- Frozen choice: `ensemble_diagnostic/FROZEN_CHOICE.json`
- Validation metrics and review: `ensemble_diagnostic/ensemble_metrics.json`, `ensemble_diagnostic/ENSEMBLE_REVIEW.md`, `ensemble_diagnostic/ENSEMBLE_REVIEW.json`
- Exploratory historical-test metrics and report: `ensemble_diagnostic/EXPLORATORY_TEST.json`, `ensemble_diagnostic/ENSEMBLE_TEST_REPORT.md`
- Uncertainty sensitivity: `ensemble_diagnostic/GROUP_BOOTSTRAP_SENSITIVITY.json`, `ensemble_diagnostic/group_bootstrap.py`
- Frozen evaluation code: `ensemble_diagnostic/frozen_test.py`

### Round04 — architecture screen (reviewed; launched, active plan)

The screen tests four arms under a common native-f objective: original tensor route, direct positive f head, independent positive trace head, and retained-residual design. A completion receipt confirms the retained-residual arm finished its fixed 20 epochs on guarded GPU5. Its selected checkpoint is epoch 0 with validation raw native-f R² 0.4052941298, numerically tied to eta0 (ΔR² 1.15e-8); epoch 20 ends at 0.3049378537. This is no selected single-model gain, and the continued-training decline is an arm-level observation, not a matched architecture conclusion. The receipt says the matched original control was still pending; the complete four-arm screen remains RUNNING/incomplete. Do not promote a result or update benchmarks from this partial evidence. Separately, legacy chan64 G1 completed naturally at epoch 227 (best epoch 75); its reassignment receipt records GPU1 passed to the independent_trace arm. These are completion/reassignment records, not current-health claims for other arms or PIDs; refresh runtime state at the scheduled monitor check.

- Design and review: `architecture/ARCHITECTURE_PROTOCOL.md`, `architecture/ARCHITECTURE_CODE_REVIEW.md`, `architecture/EXECUTION_REVIEW.json`
- Configuration and preflight: `architecture/config.json`, `architecture/PREPARATION.json`, `architecture/PREFLIGHT.json`, `architecture/MODEL_PREFLIGHT.json`
- Queue and results to monitor: `architecture/ARCH_QUEUE_STATUS.json`, `architecture/logs/`, `architecture/runs/<arm>/status.json`, `architecture/runs/<arm>/FIT_COMPLETE.json`, `architecture/runs/<arm>/INVALID.json`, `architecture/runs/<arm>/FAILED.json`
- Completed-arm receipt and interpretation: `completion_receipts/ARCHIVE_MANIFEST.json`, `completion_receipts/RETAINED_RESIDUAL_COMPLETION.json`, `completion_receipts/RETAINED_RESIDUAL_SCIENTIFIC_INTERPRETATION.md`, `completion_receipts/G1_COMPLETION_REASSIGNMENT.json`
- Bounded baseline feasibility note: `DETANET_SCALED_BASELINE_FEASIBILITY.md`; it clarifies that eager test-array loading alone is not evidence that test labels affected training or selection.

Deferred diagnostic: after matched arm summaries and resource review, consider one frozen-checkpoint validation replay comparing base_f with abs(base_f+delta_f) and error concentration by S7/S8 and fixed q90 complement. This is diagnostic only; start no new architecture or training until the four-arm screen is complete.

## Resource and stop policy

The user authorized architecture changes and broader GPU use. Current evidence-based pool policy is: keep GPUs 3 and 7 excluded for uncorrectable ECC/pending remaps; use clean GPUs 1, 4, and 6 when free (GPU2 is currently allocated to the replacement loss pilot); allow GPU5 only for guarded runs with a preflight numerical check and stop-on-new-fault behavior; preserve unrelated SpecGPT on GPU0. GPU2 is allocated to the replacement loss pilot queue (PID 785736) after the reviewed G2 stop handoff.

G2 was gracefully administratively stopped at epoch 215, cursor 114496/120355, after the reviewed queue amendment passed. The resumable last checkpoint hash is `f437e1ee9a0f569cc6b788e5df12ad3278d28fa867853e5067fca7f71dce7cfa`; the retained best hash is `1894f778b2d98a99cc390237c0516d98279375f140ad4b5ea72ac8d08012bd1a` (hash metadata only; checkpoint files remain on the server). The original supervisor is expected to record `STOPPED_REQUIRES_EXPLICIT_RESUME` / `CAMPAIGN_BLOCKED`; this is an intentional incomplete campaign, not a fit completion or unexplained failure. No `FIT_COMPLETE` was forged. Other chan64 runs continue. After they finish, analyze the campaign under an explicitly partial/administratively-stopped label.

## Monitoring, publication, and provenance

The four-hour heartbeat `qm9s-e-a` is ACTIVE. It covers the chan64 workers, the GPU2 loss pilot queue, and the architecture queue. The next scheduled check is **2026-09-29T02:29:29+08:00**. Use the current PIDs and paths recorded by the scheduled check; do not infer runtime state from an old snapshot.

Local D: archives were hash-verified from USTC-A800 before publication staging:

- `D:\MTO\archives\oscillator_r2_20260928\round01_calibration\MANIFEST.sha256.json`
- `D:\MTO\archives\oscillator_r2_20260928\round02_continuation\MANIFEST.sha256.json`
- `D:\MTO\archives\oscillator_r2_20260928\round03_ensemble\MANIFEST.sha256.json`

Publication to `techandscixie2005/MTO-1` via SSH is **pending Git transport**. SSH authentication succeeded, but a branch push has not been recorded. See the D: `PUBLICATION_STATUS.md` for the exact transport attempts. Upload only reviewed lightweight material from the verified D: copies; exclude checkpoints, optimizer states, caches, raw tensors, and other large runtime artifacts.

## Publication and targeted resource update (2026-09-29T00:13:53+08:00)

The preceding publication paragraph records the earlier archived snapshot. GitHub SSH now confirms branch `codex/oscillator-r2-research-20260928` at commit `25f208aee122ad09e5737a737e4fcef993fab954`, based on `main` `1de09d232379eb05a999cb67f58f084476b107b6`. The first commit contains the 70 verified lightweight files staged from D: archives; its diff adds only `research/oscillator_r2_20260928/`. The GitHub integration declined draft PR creation with HTTP 403, so no PR is recorded. The D: publication status holds transport and hash details.

A targeted resource check, separate from the four-hour monitor, found original chan64 G4 naturally `FIT_COMPLETE` by early stop at epoch 223 (best validation objective 0.1363691372 at epoch 71; 152 non-improving epochs). G4's oracle-E validation f R² 0.3483350041 is diagnostic only: that model does not produce deployable native f. Its former PID 566532 is gone; clean GPU6 was assigned to architecture original control PID 790200. No G4 retirement signal, GPU reset, or test inference was performed. See `G4_COMPLETION_REASSIGNMENT.json` and `.md` for the reviewed hashes and receipt. The scheduled four-hour monitor remains due at **2026-09-29T02:29:29+08:00**; this targeted check does not replace it.
