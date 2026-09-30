# MTO single-model research — current coordinator handoff

Updated 2026-09-30 Asia/Shanghai. Research is unfinished. Round03 has completed; no new best model is established. All production fits and the authorized affine evaluation are terminal. Do not relaunch them.

## Immediate work and authority

1. Round03 final report, independent integrity/scientific reviews and root decision are complete and published. The final review manifest is SHA25653a3d1366c4b06c7186550701353527ffbd9ea8874f92e2aee9440bb86a5048c; frozen decision SHA25610afc8f99dfdf04e27a1a9277118cb2f5a04b60ef2e523af9ee833d5bf7c485a.
2. Required D-first archive and inspected publication are COMPLETE at commit `4f9ae50647f957ac7d68ab1a20200a79fa527cf9`, parent `1b34e839335177e1fd582944549be4f181dee3c0`. Receipt: `ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json` (SHA256ffac1aa2c625ecd2b3e933494eb84f0e8b99efc063f94fdf31b9343144b85c1c). Preserve the published snapshot; current operational status was updated only afterward.
3. The 2026-09-29T23:50:29Z heartbeat advances Round04 preparation under `ROUND04_PREPARATION_DECISION.md`. After independent protocol review, implement and test the fixed four-coefficient f-only map using synthetic fixtures, read-only calibration-cache contract/rank checks and CPU one-checkpoint synthetic-geometry checks. No real-data fit, validation scoring, test access or new experiment is authorized. The previous heartbeat's implementation restriction applied to Round03 closeout; completed records remain unchanged.

The accepted preparation proposal is a matched pair of shared four-coefficient piecewise-linear f-only maps, using fixed TRAIN q90/q99 knots and the two frozen Round03 prediction sources. Direct FP64 least squares replaces the contemplated MLP/optimizer/epoch search. Existing frozen affine maps provide same-input linear controls. The precise protocol, failure behavior, source contracts, preflight, independent review and archive/publication must pass before any separate execution decision. Adding E or other context would require same-input linear controls. Scratch F/decorrelation and AO architecture work remain deferred.

## Round03: completed affine transfer diagnostic

Server: `/home/inspur/MTO-1/research/single_model_20260929/round03_transfer`.

| Fixed deployed recipe | Validation pooled raw-f R² |
|---|---:|
| Native full eta0 | 0.405294124 |
| Affine from full-baseline in-sample predictions | 0.402024351 |
| Affine from held-out-source predictions | 0.416374515 |
| Historical validation-fitted affine | 0.418119244 |

The held-out map gains 0.014350164 over the in-sample map and 0.011080390 over native. Both clear the predeclared 0.003 preparation threshold. It trails the incumbent by 0.001744729; retain the incumbent. Eight states improve versus native; S1/S5 regress. MAE rises from 0.017785608 to 0.017805840 and TRAIN-q99 RMSE from 0.219912917 to 0.227281574. Relative to the historical calibrated incumbent, the transferred map improves MAE/tails but loses pooled R². All labels remain included.

A fresh original-MTO source trained on 96,284 original-TRAIN molecules for exactly 33 epochs / 49,665 updates. Its initialization statistics used only those fitting rows. The remaining 24,071 molecules / 240,710 valid printed-f labels were identical for both coefficient fits. All 69 frozen hashes, 33 data orders, optimizer steps, final/last tensors and receipt bindings passed review. Coefficients froze before the single 66,860-label outer-validation evaluation. Each export contains the SAME original full baseline plus one fixed map; the temporary source is absent at deployment.

Held-out map: max(0, 0.9025853252242239 * native_f + 0.0018285582112617521). It replaces any prior affine; do not compose maps. Its temporary source scored 0.429408124 on the calibration rows, versus 0.661066543 for the full baseline that trained on them. Training size, source quality and update history differ; this does not isolate membership as a cause. One internal holdout is not full cross-fitting or fresh outer confirmation.

Source process exit and GPU1 execution were verified. The affine-fit interval included an unidentified short-lived MTO PID on GPU0; complete fit placement could not be established. Preserve `RESOURCE_PLACEMENT_NOTE.md`. The fit was not repeated. Validation was prebound to GPU1's UUID before Python startup and its owned PID was observed only on GPU1. No unrelated process was signaled or modified; performance impact was not measured.

Authoritative terminal records: `runs/source33/FIT_COMPLETE.json`, `affine/COEFFICIENTS_FROZEN.json`, `affine/EXPORT_COMPLETE.json`, `affine/VALIDATION_COMPLETE.json`, `ROUND03_RESULTS.json/md`, `INDEPENDENT_TERMINAL_INTEGRITY.json`, and `INDEPENDENT_SCIENTIFIC_REVIEW.md`. Use actual receipts for checkpoint hashes/paths. Never repeat a completed stage. The old runtime handoff is a historical execution/recovery reference, not permission to rerun.

Frozen manifest: 93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37. Prelaunch review: a6807aae55c59f254f2048b1d696a754f130075e32ef4763dfa1c94f4fb1c04c. Preparation publication: 1b34e839335177e1fd582944549be4f181dee3c0.

## Strongest eligible recipe

Native eta0 seed11 epoch33: validation R² 0.405294118341; reused historical-test R² 0.455815109310. Checkpoint `/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`, SHA 9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136.

Strongest recorded eligible recipe: same model with max(0, 0.8511830211044088 * native_f + 0.003725185373211049). Historical OOF validation R² 0.41665766, full-validation refit/replay 0.4181192453, reused historical-test 0.459667233057. Calibration worsens native MAE/bright tails; historical test improvement interval spans zero. OOF also inherits prior checkpoint selection. E/A remain auxiliary and do not exactly reconstruct calibrated f.

Self-contained geometry-only checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`, SHA bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9. Commit 69bf39f's F5 is a five-model ensemble and ineligible. No model/checkpoint prediction averaging.

## Completed negative evidence and remaining hypotheses

- Round01 full-model control/F/raw-M-decorrelation/both: all completed 20 epochs and selected epoch0. F was active and decorrelation reduced overlap. No promotion. False-bright errors outweighed improvements elsewhere; this was a descriptive reused-validation diagnosis, not a proved cause.
- Round02 frozen right-CG F: trace selected epoch2 at 0.4054078302, raw-f selected epoch1 at 0.4053875246. Both tiny gains failed the 0.003 threshold and both fixed-calibrated scores lost to the incumbent. No extensions or seed confirmation. F moved about 5–6%; inert parameters do not explain the result.
- Earlier h-only residual, direct-f/scratch objective, width and continuation studies are documented in `history_baseline.md` and the completed reports. Do not repeat unchanged settings. Audited history contains no matched scratch right-F/decorrelation factorial; that is a separate future decision.
- `NEXT_DIRECTIONS_AFTER_ROUND03.md`, its independent review, `NEXT_DIRECTION_FEATURE_CLARIFICATION.md`, and `CALIBRATION_HISTORY_CHECK.md` document the next question. Earlier calibration tested scale/per-state shrinkage/affine maps, not a flexible f-only map. Hidden h/M features do not share an identified basis across independently trained models.
- AO integrals can be computed from geometry plus an explicit basis without MO/X/Y, but the transition density remains missing. Onsite s-p terms remove the nuclear-span obstruction of point-charge covariance. Existing direct PSD outputs already span those tensors; no accuracy benefit is established. AO fitting is deferred. The six-file feasibility manifest is 82e09937178c5474db09cd5b2890189fb4d4e857f887cc58262a474ac8059172.
- Available QC labels include E and length/velocity/magnetic transition moments and strengths. No audited full X/Y, MO coefficients or transition densities. Scalar reconstruction cannot establish vector frame/phase alignment. Shared Input-frame provenance is supported; direct alignment remains unverified. Current interference/decorrelation interpretations are empirical hypotheses, not identified quantum operators.

## Evaluation invariants

Frozen outer split: TRAIN 120355 / validation 6686 / historical test 6686, ten valid printed-f labels each including zeros. Pooled R² flattens every valid scalar, not an average of state R². TRAIN bright thresholds are 0.0549 / 0.2412. Older 0.0546 / 0.2377 cuts were validation-derived and must be labeled as such.

Split SHA 141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca; dataset SHA be8fadca203429575d70642b617730592693be40858dca7189c098a671596330. No verified fresh holdout. No score-driven exclusions or test-driven selection. Independent seeds are training replication, not fresh data; report each predictor separately. Legacy array containers may include test rows, but the pilots never index them for fitting/inference/scoring/selection.

## Monitoring and ownership

Dedicated monitor: `/root/monitor`; science: sole experiment/analysis execution owner; history: independent review; root: scientific decisions. Subagents do not change goal lifecycle.

Existing ACTIVE app heartbeat `qm9s-e-a` runs every four hours on this chat. Existing read-only server cron invokes campaign `ops/monitor.py` every four hours. Preserve both, create no duplicate, and stay quiet when unchanged/non-actionable. Monitor only explicitly registered owned experiments using PID/start/boot/cwd/argv/GPU UUID, never a stale PID alone. An agent interruption or missing observation is not a training failure.

Preserve unrelated GPU0 SpecGPT. Exclude GPUs3/7 with uncorrected ECC. Preferred GPUs1/2/4/6 require immediate health/occupancy/UUID admission and the shared GPU lock; GPU5 is reserve. Prebind the intended physical GPU UUID before Python imports, and record actual owned process/device placement. Do not reset GPUs or signal unknown jobs. Server and Windows clocks differ by about137 seconds; establish ordering through receipts.

## Archive and publication

Server campaign `/home/inspur/MTO-1/research/single_model_20260929` is not a Git checkout. Preserve dirty local cwd `C:/Users/master/Documents/ChatGPT/MTO` and publication repo `D:/MTO/publication/MTO-1` (unborn worktree, preexisting metadata). Monitor owns the proven separate-index/no-filters/byte-verification publishing workflow and scoped authenticated SSH proxy; no global config or force push.

| Completed archive under D:/MTO/archives/single_model_20260929 | Verified commit |
|---|---|
| round00_audit | 0e612523cd99594b6ce05b491b8de1b18d8a2e66 |
| round01_factorial_complete | 2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae |
| round02_frozen_preparation | a4c2fabd837bea5ae7eced79d11de797af3900bf |
| round02_frozen_complete | bdc28c351b941a58dae49f833284e13cf5fdc46b |
| round03_transfer_preparation | 1b34e839335177e1fd582944549be4f181dee3c0 |
| round03_transfer_complete | 4f9ae50647f957ac7d68ab1a20200a79fa527cf9 |

Round03 completion archive/publication is COMPLETE: `round03_transfer_complete`, 167 server records /1,858,796 bytes;169 staged blobs /1,890,382 bytes independently byte-verified. Bundle SHA256d3d6c9097e0cd166fc0760f34d95af5a2b2faa95c0bf21246b0982c65b9f0178; verified tree18bbef20868087fb6631371680ba700bd03aaed6. The archive includes final reports/reviews, root decision, source/affine logs, supplemental feasibility/history notes and exact uncertain fit-placement evidence. All tensors/arrays remain server-only. Any subsequent publication must use the current remotely verified descendant of4f9ae506, without force or rebase.

EVERY completed/failed round: FIRST download lightweight records to D archives and verify all hashes; THEN inspect an explicit staged allowlist, commit, non-force push and verify remote. Never upload/download for publication any checkpoint, weights, optimizer state, raw dataset, identity/split arrays, prediction arrays, caches or credentials. Keep resumable checkpoints server-side and preserve honest failure/null results.
