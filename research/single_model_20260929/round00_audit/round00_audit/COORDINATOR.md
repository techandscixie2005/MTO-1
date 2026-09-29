# MTO single-model research coordinator

Updated: 2026-09-30 (Asia/Shanghai; campaign namespace remains 20260929). Status: architecture preflight and independent architecture review PASS; runner implementation/review and round00 publication pending, no fit launch yet.

## Objective and hard constraints

Improve pooled raw oscillator-strength R² with one model and one checkpoint at inference. No ensembles, checkpoint averaging, selective exclusions, test-driven tuning, or unavailable quantum-chemical inference inputs. Preserve frozen splits and valid labels. Checkpoints are selected using validation. Report pooled and per-state raw-f errors plus bright-tail errors. Historical-test reuse is not fresh holdout confirmation. Physical readings of learned representations remain hypotheses.

Server: `ssh USTC-A800`, repository `/home/inspur/MTO-1`.
Local workspace: `C:\Users\master\Documents\ChatGPT\MTO`.
Archive root: `D:\MTO\archives\`.

## Ownership

- Coordinator `/root`: strategy, prioritization, acceptance gates, decisions, and handoffs.
- `history_baseline`: historical evidence including commit `69bf39f`, strongest valid single-checkpoint baseline, split and evaluation audit.
- `science_implementation`: architecture and QC-label audit; later isolated implementation, correctness tests and controlled experiments after coordination.
- `monitor`: dedicated job/GPU health, persistent four-hour monitoring, scoped recovery and archive/publication operations.

Agents must preserve concise handoffs in this directory. Do not mutate another agent's owned files or interfere with unrelated server jobs.

## Research sequence

1. Establish the strongest verified baseline, its checkpoint, validation-selection rule, exact training settings and prior failures. Inspect server health and available resources before launch.
2. Prepare a matched 2×2 pilot: original architecture; shared nonlinear equivariant right-operand adapter F_theta before CG(M0,F_theta(Mk)); weak normalized decorrelation of untransformed state representations; both. Make identity initialization, equivariance, gradients, valid-state masking, and resumability explicit. Freeze pilot settings and promotion rules before reading pilot outcomes.
3. Promote only scientifically supported changes using validation. Confirm with independent seeds, reporting individual models without prediction averaging. Keep the strongest historical model available throughout.
4. Audit transition-amplitude/density and explicit-dipole reconstruction before any label-dependent auxiliary objective. Handle full TDDFT X/Y, units, state phases, degeneracy and symmetry. Consider NTO electron–hole representations, coherent interference and shared response operators when supported by audited labels and evidence.
5. Deliver one verified recipe, reproducible commands, server checkpoint path, matched comparisons and limitations. Do not describe launches or partial validation improvements as completed accuracy gains.

## Records and publication order

For EVERY completed experimental round, FIRST download lightweight server records to `D:\MTO\archives\`; verify the downloaded files; THEN stage an explicit allowlist, inspect staged names/content/size, commit and push. Include objective, exact settings, code change/source identity, logs, metrics, analysis and next decision. Never commit or upload checkpoints, weights, optimizer state, raw datasets, caches, credentials, or unreviewed unrelated files. Persist archive and commit/push receipts.

## Durable monitoring

Existing app heartbeat `qm9s-e-a` reused and updated ACTIVE on 2026-09-29, every four hours, targeting current chat `01a0edd8-8c03-75d1-aeb0-953f11af6433`; monitor independently verified settings. It delegates checks to a dedicated monitoring subagent and remains quiet when nothing actionable changes. Server cron `0 */4 * * *` invokes scoped `research/single_model_20260929/ops/monitor.py`; manual smoke and 11 safety tests passed. Actual timed trigger verified at 2026-09-30 00:00:02+08:00 in `monitoring/CHECK_20260929T160002Z.json` and `ops/CRON_TRIGGER_VERIFICATION.json`; next04:00. The server checker records health and failures without process signals; the dedicated agent handles scoped diagnosis/recovery. Explicit registry is initially empty.

## Decisions so far

- Audit before launching: no new training authorized to agents until baseline/settings/resource findings are available.
- Use frontier reasoning agents for history reconciliation, equivariant architecture and cautious server operations; simplify later isolated tasks only where appropriate.
- No assumption that quantum-state orthogonality justifies orthogonality of arbitrary learned features.
- New improvements require validation evidence and independent-seed confirmation; reused historical test is clearly labeled.
- Commit `69bf39f` final F5 is a five-model ensemble and is ineligible. Native-network eta0 reference: validation pooled raw-f R² 0.405294118; reused historical-test R² 0.455815109. Checkpoint `/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`; SHA256 `9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136`. Calibrated/residual single-model recipes must also be reported and assessed, not silently excluded.
- Server tree is an artifacts/source workspace without `.git`. Publication repository is `D:\MTO\publication\MTO-1`; it uses Git plumbing and requires careful inspection before branch/worktree operations. Local cwd contains preexisting dirty experiments; preserve them.
- GPUs 1,2,4,6 are preferred pending launch-time checks. GPU0 is occupied by an unrelated SpecGPT job. GPUs3 and7 have uncorrected ECC and are excluded. GPU5 is reserve because of corrected-ECC history.
- Adapter implementation authorized only in isolated `/home/inspur/MTO-1/research/single_model_20260929`: right operand of CG for excited states k=1..10; M0 and existing invariant/raw-2e skips unchanged; shared equivariant scalar-gated residual with zero-initialized final mixing. This is an effective readout hypothesis; current output is PSD tensor C Cᵀ, not an explicit transition dipole vector.
- Weak normalized decorrelation uses raw untransformed M: concatenate 0e, 1o/sqrt(3), 2e/sqrt(5), normalize each excited-state representation, then average squared cosines over valid off-diagonal excited-state pairs. Exclude M0 because shared-reference independence is a separate unsupported constraint. Lambda candidate 1e-3 awaits a train-only loss/gradient scale check and must then be frozen.
- First factorial: all arms warm-start exact eta0 epoch33, original LE+Ls loss, reset optimizer identically, fixed LR 1e-5, batch64, seed/data order11, FP32, AMSGrad, clip5, weight decay0, 20 epochs, epoch0 eligible. Historical 1e-4 continuation controls failed; 1e-5 is a new stability-preserving schedule hypothesis that requires its own matched control. Do not attribute earlier failure to learning rate without evidence.
- Pilot checkpoint selection: maximize pooled validation raw-f R² on all valid labels. Fix bright-tail thresholds from training labels before outcomes. No test evaluation during pilots. Report all four arms including failures and epoch0 selections.
- Pilot promotion decision is validation-only: prioritize a change gaining at least 0.003 absolute pooled raw-f R² over both the matched control and its epoch0 baseline. Smaller gains are inconclusive and may justify a specifically motivated follow-up, not an accuracy claim. Inspect per-state and bright-tail tradeoffs, compute cost and stability before promotion. This is a resource-allocation threshold, not a statistical significance claim.
- Confirm promoted recipes against matched controls using independent seed models or fresh independent training. Multiple continuations of one checkpoint only establish sensitivity to continuation randomness and must not be described as independent full-training seeds. Never average predictions.
- There is no verified fresh holdout. All 133727 curated molecules are assigned to frozen train/validation/test; remaining158 source records have zero states, failed convergence/termination and missing geometry. Do not use them as holdout or change the frozen split.
- Extracted labels contain energies, printed raw-f, length transition dipoles/A, velocity and magnetic dipoles; no available MO coefficients, AO dipole integrals, full TDDFT X/Y or transition densities. Raw source logs are unavailable on this server. Explicit transition-density/NTO reconstruction is therefore unsupported without separately obtained/audited QC data. Full TDDFT length response involves X+Y; an X-only NTO implementation is not full-response reconstruction.
- Current PSD decoder already contains coherent cross-channel interference before taking norms. Do not claim PSD inherently loses all interference. A later geometry-only shared symmetric response/Hamiltonian plus PSD cross-state strength operator is a possible inductive bias; learned operators would remain unidentifiable from E/f alone. Synthetic gauge, degeneracy and centrosymmetry tests are required before any such pilot.
- Round00 publication is authorized after prelaunch review PASS: archive locally first, then explicit inspected commit on `codex/single-model-20260929`, based on verified remote main, without force push. Monitor owns safe plumbing-repository publication and receipts. No fit launch before correctness review.
- Architecture preflight and independent architecture review PASS. Adapter adds4016 active parameters (0.259% over1552092 baseline); identity-initialized E/A/f are bitwise baseline-equivalent; nonzero adapter O(3) checks include reflection. Full wrapper FP32 errors are within reported tolerances. Zero/one valid states, masked NaNs, zero vectors, phase/scale invariance and exact Adam next-update resume tests passed. Lambda1e-3 is frozen after four train-only batches: weighted penalty/base gradient ratio0.000369–0.00168.
- Science agent owns runner, configs, training and metrics; launch is authorized AFTER independent runner review PASS and archive/push receipt, plus immediate healthy/free GPU checks and registry registration. All arms reset runtime RNG identically after model construction and use independent data-order generators. Atomic epoch-boundary last checkpoints include optimizer/RNG/order; mid-epoch failure restarts from prior complete epoch. Epoch0 full-validation replay is mandatory (absolute raw-f R² difference<1e-5; explain any tolerance).
- Precise isolation statement: legacy Data loads full source array containers containing test rows, but pilot fit/evaluation indexes only train and validation. No test inference, scoring, or use in fitting/selection. Do not claim test data files are never loaded.
- Archive includes a small explicit snapshot of legacy source dependencies with original path/SHA mapping and runtime versions; model checkpoints and dataset arrays stay on server. GPU admission rejects active compute occupancy/unhealthy devices while permitting tiny unrelated graphics allocations.
- Strongest recorded eligible single-checkpoint baseline is now packaged and replay-verified: `baselines/calibrated_eta0.pt` under campaign server directory, SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`. It stores the entire eta0 model and two fixed buffers, no optimizer. Formula `max(0,0.8511830211044088*f+0.003725185373211049)`. Full-validation replay R²0.4181192453; this is validation-refitted performance, not independent evidence. Historical OOF score0.41665766 and reused-test0.459667233 remain separately labeled. Cross-device formula replay tolerance verified; E/A stay auxiliary and no longer reconstruct calibrated f exactly.
- BEFORE pilot outcomes, prescribe secondary metrics applying these SAME fixed calibration constants to every arm's validation predictions, with no refit/search. Primary checkpoint selection remains native raw-f R². This provides fair comparison to the strongest eligible recipe; beating0.4053 native alone does not establish superiority to the calibrated recipe. Include pooled/per-state/bright-tail tradeoffs, calibration exposure caveat, and auxiliary-output consistency caveat.

## Next coordinator actions

Receive three audit handoffs, settle baseline and evaluation contract, specify pilot duration/seeds/loss strengths from evidence, authorize implementation and matched launches, then archive the initial audit/code round before publication.
