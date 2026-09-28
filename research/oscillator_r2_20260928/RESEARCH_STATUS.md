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

### Round04 — architecture screen (completed; validation-only results awaiting scientific review)

The four-arm screen completed its fixed 20 epochs on the frozen validation split. The queue terminal receipt is `architecture/ARCH_QUEUE_STATUS.json` (`ALL_COMPLETE`, exit code 0); result summary files are `architecture/ARCHITECTURE_RESULTS.md` and `.json`. Validation best raw native-f R²: original .405294 (epoch 0), retained_residual .405294 (epoch 0; numerical tie), direct_f .369250 (epoch 1; Δ−.036044), independent_trace .372023 (epoch 1; Δ−.033272). Bootstrap positive fractions for direct_f and independent_trace were 8.75% and 8.65%; neither approaches the promotion rule. Retained-residual final epoch 20 fell to .304938, while its selected checkpoint remained epoch 0. No test evaluation occurred. These are validation results awaiting independent scientific review; do not promote a model or change the project benchmark from them. Separately, legacy chan64 G1 completed naturally at epoch 227 (best epoch 75), and its completion/reassignment receipt records GPU1 passed to independent_trace. The queue and four arms are complete; loss-pilot status remains on its own scheduled check.

- Design and review: `architecture/ARCHITECTURE_PROTOCOL.md`, `architecture/ARCHITECTURE_CODE_REVIEW.md`, `architecture/EXECUTION_REVIEW.json`
- Configuration and preflight: `architecture/config.json`, `architecture/PREPARATION.json`, `architecture/PREFLIGHT.json`, `architecture/MODEL_PREFLIGHT.json`
- Queue and results to monitor: `architecture/ARCH_QUEUE_STATUS.json`, `architecture/logs/`, `architecture/runs/<arm>/status.json`, `architecture/runs/<arm>/FIT_COMPLETE.json`, `architecture/runs/<arm>/INVALID.json`, `architecture/runs/<arm>/FAILED.json`
- Completed-arm receipt and interpretation: `completion_receipts/ARCHIVE_MANIFEST.json`, `completion_receipts/RETAINED_RESIDUAL_COMPLETION.json`, `completion_receipts/RETAINED_RESIDUAL_SCIENTIFIC_INTERPRETATION.md`, `completion_receipts/G1_COMPLETION_REASSIGNMENT.json`
- Bounded baseline feasibility note: `DETANET_SCALED_BASELINE_FEASIBILITY.md`; it clarifies that eager test-array loading alone is not evidence that test labels affected training or selection.

Deferred diagnostic: after independent review and resource review, the coordinator may consider one frozen-checkpoint validation replay comparing base_f with abs(base_f+delta_f) and error concentration by S7/S8 and the fixed q90 complement. This is diagnostic only; the result summary alone does not authorize another architecture launch.

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


## Round 04 final review and validation replay (2026-09-29)

The four-arm architecture screen is complete and independently reviewed. No candidate is promoted; selected checkpoints for direct-f and independent-trace are below the unchanged original, while retained-residual epoch 0 numerically reproduces eta0. The screen used validation only. See `architecture/ARCHITECTURE_RESULTS.md`, `analysis_addendum/ARCHITECTURE_VALIDATION_ADDENDUM.md`, and `completion_receipts/ARCHITECTURE_SCIENTIFIC_DECISION.md`.

The authorized final20 replay is complete and independently reviewed, validation only. Raw native-f validation R² is 0.4052941183 for frozen eta0, 0.2789672525 for the changed base, and 0.3049378477 for emitted final20; emitted final20 adds 15.8971 SSE over eta0. This is an inference decomposition of one jointly trained checkpoint, not a retraining counterfactual. The largest deterioration molecule accounts for 72.97% of net SSE increase and the top 10 for 97.27%. No test split was read, and runtime arrays remain server-only. See `completion_receipts/FINAL20_REPLAY_RESULTS.json` and `completion_receipts/FINAL20_REPLAY_INDEPENDENT_REVIEW.md`.

A single frozen-eta0 residual-head-only 20-epoch diagnostic is authorized for preparation under `frozen_residual/FROZEN_RESIDUAL_PROTOCOL.md`, subject to implementation review, preflight and resource admission. No training launch is recorded yet. The round remains validation-only; no model or benchmark is promoted.


## Round 05: frozen residual-head result (2026-09-29)

The fixed 20-epoch, train/validation-only run completed 37,620 optimizer steps. It selected epoch 2 at pooled raw native-f validation R² 0.406789903, a +0.001495785 change from eta0 (0.4052941183); epoch 20 was 0.404299111. The independent run audit and scientific result review passed. This small one-seed validation gain is below the +0.01 promotion threshold, so no model or benchmark is promoted. No test data were used and no extra epochs are authorized by this result. See `frozen_residual/ROUND_REPORT.md` and `frozen_residual/SCIENTIFIC_RESULT_REVIEW.md`.

## Round 06: frozen-Gram probe (preparation only)

The bounded two-arm protocol is approved at `frozen_gram_probe/FROZEN_GRAM_PROBE_PROTOCOL.md` (SHA-256 `c7f6f6ede884e663d5af579e6add91f49552128ff39ce7dc0bdb73088777fa32`). Implementation and preflight remain pending; no full-fit launch is recorded. Launch is conditional on independent review and runtime admission gates.


## Scheduled check and completed continuation pilot (2026-09-29 02:29 +08)

The replacement matched loss pilot is complete. Control, weighted and direct-f-matched all selected epoch 0 at native-f validation R² 0.405294108, 0.405294109 and 0.405294128. These are numerical ties; epoch-20 scores declined, and no test or promotion was used. See `pilot_summary.json`.

Legacy G3 remains active on GPU4 at epoch 281/cursor 108800; its status does not expose native-f R². G1 and G4 completed naturally. G2 remains an intentional administrative stop with the original campaign partial.

The completed frozen residual head-only run selected epoch 2 at R² 0.406789903 (+0.001495785 vs eta0) and finished at 0.404299111. The gain is below threshold; no promotion or test. The separate frozen-Gram probe failed before fitting at the runtime feature identity gate (`h` max difference 2.384185791015625e-06); it has no model result and awaits reviewed repair. The approved eta0 seed protocol is prepared, with no training launch recorded at this check.

Next scheduled full monitor: 2026-09-29 06:29:29 +08. GPU2 and GPU6 were idle and clean; G3 owned GPU4 with no ECC errors. GPUs3/7 remain excluded and GPU0 remains reserved for the unrelated job.


## Receipt update (2026-09-29 03:00 +08)

The completed loss-pilot scientific review is available at `completion_receipts/LOSS_PILOT_SCIENTIFIC_REVIEW.md` (SHA-256 `5ba23d7a312cea6e101fef67737632307f617de4140ff4c57906481c40abaebc`). It confirms the epoch-0 variants are numerical ties; no promotion, test, or extension.

Two fresh-initialization eta0 seed replications, each fixed at 100 epochs, were launched from reviewed protocol/review gates: seed 23 on GPU1 (PID 859155; receipt SHA-256 `af79442051f60dc0fd29e11c670ca3949237f312323f09a0d2b2829a8dfdfa3d`) and seed 37 on GPU6 (PID 859334; receipt SHA-256 `789d6b8fb4f5390b2313e33a2e107a915aa8c560ddf5e2fae32892f1500ed879`). These are launch receipts, not current health claims.

Corrected frozen-Gram attempt 2 passed independent review and launched on GPU2 (receipt SHA-256 `8481aa4ab882a7d87ed42c9963948523614d050bbafe947520253bc74ce95021`). Its scientific evaluation completed, but report writing failed under ASCII encoding; an additive audited report recovery is underway. Attempt 1 remains preserved as a pre-fit cache-gate failure. No refit or new inference is authorized by this status note.

Next scheduled full monitor remains **2026-09-29T06:29:29+08:00**.


## Additive review and launch receipts (2026-09-29 03:00 +08)

The loss-pilot scientific review is finalized (`completion_receipts/LOSS_PILOT_SCIENTIFIC_REVIEW.md`, SHA-256 `5ba23d7a312cea6e101fef67737632307f617de4140ff4c57906481c40abaebc`): epoch-0 variants are numerical ties; no promotion, test, or extension.

Frozen-Gram attempt 2 completed its saved numerical evaluation. Its original report writer failed on ASCII encoding; additive report recovery and independent postprocess/scientific reviews are complete. The original failure remains provenance, and there was no refit or new inference. See `frozen_gram_probe/SCIENTIFIC_RESULT_REVIEW.md` (SHA-256 `cc8d3aeea23c18ba2f721f8e48e948a1cca30085230b902b599f947bd4d13eed`) and `POSTPROCESS_REVIEW.json` (SHA-256 `7d9f406ac182da6ee5cceec227f2ccf02f062cb050a4893d66877ef4225843dc`).

The reviewed eta0 seed protocol/implementation was launched as two fresh-initialization, fixed 100-epoch runs: seed 23 PID 859155 on GPU1 and seed 37 PID 859334 on GPU6. Their launch receipts and implementation review are recorded in `eta0_seed_replication/`; these are launch records, not a fresh health check.

The next scheduled full monitor remains **2026-09-29T06:29:29+08:00**.


## Scratch readout comparison launch receipts (2026-09-29T03:21:03+08:00)

The reviewed fixed-100-epoch study has two launch receipts: MTO arm PID 879890 on GPU2 (start_ticks 1247080802, receipt SHA-256 `d4bf260b73e6db099742c7bcef8be0c4834adad2202c99486b7130be0873d39c`) and native arm PID 880266 on guarded GPU5 (start_ticks 1247085838, receipt SHA-256 `cd006ac90880c3f2b7e17d5a65aa064070e8eed72f5b7d272949b0378507b837`; admission review `ce61348caab6b1367787e18eeb89e2dfe5cf91349ad4e0602d2bf0dff9d4183d`). These receipts do not claim current health. Implementation review SHA-256 `6226b1c68226ce8a5ed6ddf74c630bce32dd88e0fadc46b0de12e8e4d005c3e2`; preflight SHA-256 `7719140e4bc0fcf5566eb6e90aa0d450bb27f3e217eb17a727a06116eec02e3c`.

The geometry audit report is complete and staged locally; its descriptive findings retain every validation molecule and make no causal or label-error claim.

Next scheduled full monitor remains **2026-09-29T06:29:29+08:00**.


## Independent geometry review and scratch scope (2026-09-29T03:25:14+08:00)

`geometry_error_audit/INDEPENDENT_SCIENTIFIC_REVIEW.md` (SHA-256 `cbfff434ea9d2df1f06b713f53788f803c960cec12b38547920138bece294e5c`) independently recalculates the descriptive geometry diagnostic. State slots are zero-based (slot 6 is S7); equal3 has slightly lower absolute SSE across the six lowest-q2 molecules but a higher share of its lower full-validation SSE. There is no causal or label-error conclusion, no sample removal, and no benchmark change.

Root authorized reuse of the fixed training-derived shape bins for saved-prediction diagnostics after the seed and scratch runs complete. Do not modify active trainers or run new inference for those comparisons.

