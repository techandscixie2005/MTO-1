# MTO single-model research — coordinator handoff

Updated 2026-09-30. Research remains in progress; no new accuracy improvement has been established.

## Current state

- Round01 four-arm full-model pilot COMPLETE: control, shared right-side F, raw-state decorrelation, and both all selected epoch0. All20epochs completed normally; selected weights equal the eta0 starting model. No candidate qualified for independent-seed confirmation.
- Independent integrity/scientific reviews PASS. Completed report: `research_state/single_model_20260929/reports/SINGLE_MODEL_RESEARCH_REPORT.md`. Scientific next decision: `research_state/ROUND01_DECISION.md`.
- Round01 archived FIRST and then published: commit `2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae`, parent `0e612523cd99594b6ce05b491b8de1b18d8a2e66`, remote branch `codex/single-model-20260929` verified.
- Round02 frozen-F two-arm preparation is underway. Real-TRAIN cache/output/gradient/Adam parity, exact variance and serialized resume checks PASS. Geometry-only one-checkpoint access audit and exact final review/source closure are remaining gates at this handoff. No production fit has launched yet.

## Root execution authorization

Science may launch the two round02 arms WITHOUT another root approval once all these gates pass: independent review bound to the exact source/config closure; real-TRAIN cache/gradient/next-update/resume/frozen-state checks; one-checkpoint geometry-only access check; preparation records FIRST downloaded to D archive, then inspected/committed/pushed with monitor's verified receipt; healthy/free GPU admission with shared locks and distinct registry entries. Repair/review failed gates before launch. No silent protocol changes.

Monitor must publish round02 preparation as a descendant of the verified round01 branch head. All trained models, optimizer states, cache and prediction arrays stay on the server.

## Round02 fixed question and protocol

Can the existing shared pre-CG adapter improve readout with every original parameter and buffer frozen, and does direct raw-f supervision help this restricted correction?

- Two active arms: original LE+Ls versus LE+Lf, same4016 trainable F parameters and exact identity initialization. F acts on excited-state right Mk before CG(M0,F(Mk)); raw bypasses and coupled E/A decoder remain unchanged.
- Lf is raw printed-f MSE divided by fixed TRAIN population variance `0.002510981243894732`, coefficient1. Verified over1,203,550 valid labels including22,646 zeros. No fitted gradient rescaling.
- Adam AMSGrad, fixed lr1e-5, batch64, seed/order11, weight decay0, clip5, FP32,20epochs, epoch0 eligible. Cache rawM only because the producing network is frozen; final inference recomputes from geometry.
- No rawM penalty (constant under this freeze), clipping of targets, robust loss, selective exclusions, output caps, new E branch, test scoring or adaptive schedule.
- Select native pooled validation raw-f SSE over all valid labels. Same historical calibration constants are secondary, without refitting. Report state/tail/false-bright/E metrics and TRAIN optimization diagnostics.
- Either recipe needs >=.003 native R² versus epoch0 for independent-seed allocation. Objective-specific promotion also needs >=.003 over the matched objective control. Do not infer a nonlinear advantage without a linear control. Beating native alone does not establish superiority to the calibrated baseline.
- Historical frozen h-only raw-f residual already achieved .4067899034 (epoch2); freezing/raw-f training alone is not novel. New rationale is location before CG and restricted parameter changes. Frozen in-sample backbone representations can still generalize poorly.

## Round01 evidence and interpretation

All arms selected native R²~.405294116 (replay noise only). F changed inputs~.75–.78% and decorrelation reduced overlap, so interventions were active.

Control epoch20 pooled SSE increased6.1206. True-dim/predicted-q99-bright SSE increased10.0074; other three bins improved. S7/S8 contributed+9.3017/+1.5589 SSE; eight states improved. Largest0.1% of errors carried22.83%→31.95% of SSE. Energy error, MAE and true bright-tail RMSE improved. This is post hoc reused-validation diagnosis, not a proved failure cause.

Global S7/S8 summed-error SSE worsened, ruling out simple global anticorrelated redistribution as the full explanation. Tight-gap cancellation remains a hypothesis. Gap audit of selected epoch0 describes baseline structure, confounded by state/brightness composition; rounded .0287 diagnostic threshold differs slightly from exact TRAIN quantile. No exclusions or primary metric changes resulted.

## Strongest eligible references

Native eta0 seed11 epoch33: validation raw-f R² `.405294118341`, reused historical-test `.455815109310`.
Checkpoint `/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`.
SHA256 `9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136`.

Strongest recorded eligible recipe: same model with `max(0,.8511830211044088*f+.003725185373211049)`. Historical OOF validation `.41665766`, full-validation refit replay `.4181192453`, reused historical-test `.459667233057`. Calibration worsens MAE/bright tails; its historical test improvement interval spans zero. E/A remain auxiliary and do not reconstruct calibrated f exactly.
Self-contained geometry-only checkpoint `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`.
SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.

Commit69bf39f F5 is a five-model ensemble and is INELIGIBLE. No prediction/checkpoint averaging is permitted. Independently trained seed23/37 contracts and promotion-gated preparation exist; no confirmation launched.

## Frozen evaluation and scientific limits

- Frozen train120355/validation6686/test6686; ten valid raw printed-f labels each, including zeros. Pooled R² flattens all states, not an average of state R². TRAIN bright thresholds .0549/.2412; distinguish older historical validation-based .0546/.2377 thresholds.
- Splits SHA256 `141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca`; dataset `be8fadca203429575d70642b617730592693be40858dca7189c098a671596330`.
- No verified fresh holdout. All historical test figures are reused evidence. Legacy loader allocates containers containing test rows, but pilots never index test for fitting/inference/scoring/selection.
- Available labels: E, length/velocity/magnetic transition moments and printed strengths. No available full X/Y, MO coefficients, AO dipole integrals or transition densities. Exact NTO/electron–hole reconstruction is unsupported. TRAIN-only velocity reconstruction matches printed precision in audited sample; full coverage/phase conventions remain unaudited.
- Current PSD decoder already includes cross-channel interference. M is not an identified electronic wavefunction. F is O(3) equivariant but not established to satisfy electronic sign covariance. Raw-state decorrelation is empirical, not verified quantum orthogonality.
- Naive shared-H eigendecomposition failed synthetic sorting/degeneracy/gradient checks. Smooth factor mixing remains an untested gauge-unidentified hypothesis. No response/velocity auxiliary fit approved.

## Monitoring, resources and ownership

Dedicated monitor agent owns four-hour monitoring, safe resource checks and archive/publication. Server cron `0 */4 * * *` runs campaign `ops/monitor.py`; actual midnight trigger and11 safety checks PASS. App heartbeat `qm9s-e-a` ACTIVE every4hours on current chat, quiet unless meaningful change. Registry PID/start/boot/cwd/argv/GPU UUID is authoritative; never act on stale PIDs in handoffs.

GPU0 has unrelated SpecGPT and must be preserved. GPUs3/7 have uncorrected ECC and are excluded. Prefer1/2/4/6 after immediate admission; GPU5 reserve with corrected-ECC history. GPU2 was designated for bounded round02 preparation. Monitor must recheck before fit. Server clock is137seconds behind Windows; establish operation order by dependent receipts.

Agents:
- science_implementation: isolated code, preflight, fitting and analyses.
- history_baseline: history/QC audit, independent scientific/source review, working report.
- monitor: persistent monitoring, locks/registry, scoped recovery, D-first archives and publication.
- root: scientific choices, promotion and coordination. Subagents must not change goal lifecycle.

## Archive and repository handling

Campaign server `/home/inspur/MTO-1/research/single_model_20260929`; server is not a Git checkout. Preserve dirty local `C:/Users/master/Documents/ChatGPT/MTO`.
Publication Git plumbing repo `D:/MTO/publication/MTO-1` has an unborn worktree and preexisting untracked metadata; do not checkout/reset it blindly. Monitor uses separate index, `hash-object --no-filters`, exact new-blob verification, scoped authenticated SSH proxy, strict existing host key; no global config/credential exposure or force push.

Round00 archive `D:/MTO/archives/single_model_20260929/round00_audit`; commit `0e612523cd99594b6ce05b491b8de1b18d8a2e66`.
Round01 archive `D:/MTO/archives/single_model_20260929/round01_factorial_complete`:163 server records,2,462,550bytes; bundle `f95858ef2c0b4be769a9bd667c7d7f1c4c9e6f2eff3442c8d780979d7223a330`. Then165 byte-exact new blobs inspected and published as `2589d0f501a9d864aa9ae6449e80e3e14d1cc8ae`. Local/server publication receipts exist.

EVERY round: FIRST download lightweight objectives/settings/source/logs/results/analysis/next-decision records to D archives and verify; THEN inspect staged explicit allowlist, commit and push. Never upload weights, optimizer states, raw datasets, prediction arrays, caches or credentials. Preserve failed-round evidence and resumable checkpoints. Completed null results are outcomes, not failed jobs.

## Next actions

1. Finish round02 final one-checkpoint audit/exact review and preparation archive/push; launch authorized paired fits only after gates.
2. Monitor safely; do not alter fixed20epoch settings using interim validation.
3. On completion, analyze all selected/fixed20 metrics and training behavior, independent review, root decision. Archive FIRST, then commit/push.
4. Promote only supported gains for matched independent seed23/37 confirmation; never average predictions. Otherwise select a genuinely new controlled question using recorded evidence.

## Round02 preparation publication receipt

- All final preflight and independent review gates PASS;64-file source closure `6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e`, review SHA `38e0d09ad8d007c5ee7036a9da5fc279ec63019bbca84ee25a0cc8774f4f6452`.
- Archive FIRST `D:/MTO/archives/single_model_20260929/round02_frozen_preparation`:92 server records505594bytes; bundle `105aee47549cb4cbf0b65d293aa25030d91164f55301a82e91889ba332d6fdc8`. Then94 byte-exact blobs staged and published.
- Verified commit `a4c2fabd837bea5ae7eced79d11de797af3900bf`, parent2589d0f, same research branch. Server `ops/ROUND02_PUBLICATION_RECEIPT.json` binds archive/source/review. No cache/weights/data downloaded or uploaded.
- Science is performing fresh healthy/free GPU admission and the already-authorized paired launch. Await actual launch receipt before describing fits as running.
