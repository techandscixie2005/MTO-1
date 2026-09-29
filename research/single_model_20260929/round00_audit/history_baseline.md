# Single-model baseline and history audit

Audit date: 2026-09-29. Read-only review of server records and local Git objects; no training, model inference, new split, or checkpoint change. Owner: history_baseline subagent. Server root is `/home/inspur/MTO-1`; evidence paths below are relative to it unless stated otherwise.

## Decision summary

Use the uncalibrated eta0 epoch-33 network as the matched architecture starting point. It is the best reviewed original native MTO network on validation, but it is **not** the numerically highest eligible single-network recipe: a fixed affine calibration and a frozen learned residual have higher validation scores. Preserve all three comparison rows and their different selection/evidence status. No ensemble qualifies for the present request.

| Single-network recipe | Validation pooled raw-f R² | Already observed historical-test R² | Evidence and eligibility |
|---|---:|---:|---|
| eta0 seed11 epoch33 | 0.405294118341 | 0.455815109310 | Original network, one existing checkpoint; reliable replay anchor. |
| eta0 + fixed clipped affine calibration | OOF 0.41665766; refit-on-validation 0.418119238799 | 0.459667233057 | No averaging, one network plus two constants. Can be packaged in one checkpoint. Largest recorded single-network validation recipe, but OOF checkpoint predictions already depend on reused validation; several test calibration results were seen before the protocol. Historical-test ΔR² interval includes zero and bright errors worsen. Not fresh confirmation. |
| eta0 + learned frozen-base residual head, epoch2 | 0.406789903436 | Not evaluated | One composite network, no averaging; 4,161 extra parameters. Current saved head checkpoint excludes base and would need a verified one-checkpoint export before satisfying literal final deployment requirement. Tiny validation gain, not historically promoted. |
| chan64 G1 epoch75 | 0.40525459 | Not used to select this baseline | Original tensor readout, legacy joint-selected; essentially tied but lower than eta0. |
| eta0 seed37 epoch44 | 0.394455334 | Not evaluated | Fixed 100epoch run; selected legacy/raw-f checkpoint aliases identical. |
| eta1 seed11 epoch25 | 0.387366710059 | Existing historical metrics not used for rank | Different own-loss selection. |
| eta01 seed11 epoch33 | 0.377126668109 | Existing historical metrics not used for rank | Different own-loss selection. |
| scratch MTO epoch selected by raw-f | 0.373492948 | Not evaluated | Fixed 100epoch scaled-objective study. |
| eta0 seed23 epoch53 | 0.368708895 | Not evaluated | Fixed 100epoch run. |
| chan64 G3 epoch174 | 0.36810069 | Not used to select this baseline | P-only tensor model, legacy joint-selected. |
| scratch native direct-f | 0.343781976 | Not evaluated | Fixed 100epoch scaled-objective study. |
| original unscaled DetaNet E/f | 0.046292167706 | Available historical only | Weak raw unscaled joint-loss control; 11,625 negative validation predictions. |

At commit `69bf39fcc98a0b25242c8ecad47ce8c5f6b4385e`, F5 is explicitly the mean of eta0/eta01/eta1/G1/G3. Validation 0.487699605 and reused-test 0.508624899 are five-model results and cannot be presented as the current single-model benchmark. Equal3, seed3, F7 and descriptor/F5 mixtures are also ineligible.

## Reproducible eta0 anchor

- Checkpoint: `/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt`.
- SHA256: `9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136`.
- Source config: `experiments/qm9s_eta_Ef_20260926/configs/mto_eta0.json`, SHA256 `e1b823aa48f491b7e0eaa8cc106fa712e6ba3e3256d36e23f091af6d9c07bdf9`.
- 1,552,092 total parameters; DetaNet backbone 1,371,840; MTO channels16, query32, router/head128, ten states. Atomic numbers and coordinates at inference.
- Seed11, Adam AMSGrad, lr0.001, batch64, weight decay0, clip5, FP32, AMP/TF32 false, two CPU threads. Maximum1000/minimum200 epochs, plateau patience50, early patience150, minimum lr1e-6. Selected epoch33; natural stop epoch236.
- Objective `LE + Ls`, with `LE = mean((Ehat-E)^2)/0.5378066634062587`, `Ls = mean((trace(Ahat)-trace(A))^2)/(3*0.1324285377060015)` over valid labels. Eta0 removes traceless Q supervision but retains energy supervision.
- Original checkpoint chosen by own validation `LE+Ls`; eta variant chosen by validation spectral MSE, **not** by retrospectively choosing maximum raw-f or test score. Evidence: `reports/selection_before_test.json`.
- Prediction `f = 2/(3*27.211386245988) * Ehat_eV * trace(Ahat)`; use predicted E, not oracle E. Pool every valid molecule/state scalar before computing SSE and SST; do not average state R². Source evaluator `evaluate.py:15-19` masks and casts raw scalars to FP64.
- Validation count66,860, SSE94.205120929384, SST158.406237158112, RMSE0.0375365254225, MAE0.0177856083459. Historical test count66,860, SSE103.193193599053, SST189.628920913713, RMSE0.0392864096125, MAE0.0177159854998.

### State and bright-tail errors

| State | Validation R² | Historical-test R² | Historical-test SSE |
|---|---:|---:|---:|
| 1 | .827284 | .816740 | 2.275335 |
| 2 | .735728 | .727137 | 5.236707 |
| 3 | .590072 | .595898 | 6.161082 |
| 4 | .390116 | .329771 | 8.583764 |
| 5 | .420834 | .371479 | 8.197937 |
| 6 | .243128 | .231045 | 10.200719 |
| 7 | .049940 | .671835 | 15.827966 |
| 8 | .306930 | .351521 | 13.704716 |
| 9 | .301463 | -.188524 | 17.249889 |
| 10 | .150100 | .188732 | 15.755079 |

Final historical-test slices use TRAIN-derived q90=.0549 and q99=.2412. Eta0 q90-and-above: 6,616 labels, SSE69.105621155, MAE.0683257903; q99-and-above:647 labels, SSE40.757549673, MAE.1767403650. Earlier validation diagnostic cutoffs .0546/.2377 were validation-derived; never relabel them TRAIN-derived. Their eta0 validation bright SSE values were60.525630554/31.612563729.

Calibration is `max(0, 0.8511830211044088*f + 0.003725185373211049)`. Historical-test MAE worsens to.01813382256; validation-defined q90/q99 test SSE rises by5.237757/4.851980. Its paired historical ΔR² interval is[-.012204,+.022829]. Residual epoch2 improves validation q90 SSE by.515986 but worsens below-q90 by.279045 and S7 by.013293. These are genuine tradeoffs, not grounds to omit the recipes from comparisons.

## Frozen population and evidence limitations

`experiments/qm9s_eta_Ef_20260926/data/splits.json` uses seed20260925, fixed counts120355/6686/6686, all current133727 molecules. SHA256 `141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca`. Dataset hash `be8fadca203429575d70642b617730592693be40858dca7189c098a671596330`; normalization hash `812a7b80d6259eb6106a5a534a8358a626713ec7ee53ad992fff6fcbabe59452`; raw printed label hash `621dc8723fb6dc96681851a26e4f52fcb7747a4778e57a249a75d22f3af4c632`.

Grouping is conservative connectivity with explicit H, successful neutral bond-order/valence check, bond-order/stereo-agnostic grouping;133469 groups,247 duplicate groups;266 unresolved groups/278 records assigned train-only. All state masks valid. No verified untouched holdout is documented in the audited project. The existing test was repeatedly exposed. A subsequent check of the extraction dataset_audit.md establishes that the158 records outside frozen IDs have zero state labels, no final normal termination or response-convergence marker, and missing geometry. They cannot supply a fresh holdout. Preserve the split. New unseen confirmation requires separately obtained/audited data or a separately justified retraining study, not relabeling current train/validation/test.

Raw f is printed on a1e-4 grid. Raw-vs-A-derived f MAE2.48655e-5, RMSE2.87835e-5, max7.49881e-5; preserve original printed f for evaluation. There are25,079 printed zero labels (25,007 with positive derived value). Do not drop them. Records include E, transition dipole-derived A and raw f; this audit has not established complete amplitudes, X/Y, orbitals or AO integrals. The science implementation agent owns that audit.

## Failed routes and rationale for the new factorial

- Loss continuation A/B/C from eta0: original LE+Ls, E²weighted trace, matched-scale native raw-f; all selected epoch0 at.405294118. Full-model Adam AMSGrad fixed lr1e-4,20epochs,batch64,seed/order11,WD0,clip5,FP32. Repeating unchanged is unjustified.
- Architecture continuation at the same lr/budget, common LE+Lf/Var_train(f): original and zero-initialized h-only retained residual also selected epoch0. Direct scalar-f .369250 and independent trace .372023. Joint residual final epoch20 fell to.304938. This does not isolate a backbone limit; objective scale and information retained differ.
- Frozen h-only residual:4,161 trained parameters,20epochs lr1e-4. Best epoch2 .406789903, final20 .404299111; no extension without new mechanism.
- Fixed frozen Gram ridge: h+base arm .40442847; plus528 within-state Gram coordinates .40281935. This does not disprove useful nonlinear pre-CG transforms or cross-state operators.
- Wider MTO channels, p-only rank-one, unscaled direct-f,100epoch scratch readouts, descriptor forest and posthoc calibration did not establish a substantial confirmed single-model improvement. Seeds23/37 final training R² rose while validation degraded; overfitting matters.

Accepted root pilot: four arms control/F/orth/F+orth, common eta0 weights and original LE+Ls;20epochs fixed lr1e-5, all other simple optimizer settings matched, epoch0 eligible, validation-only raw-f selection. Shared equivariant nonlinear F on right Mk before CG tests a new interaction location rather than repeating a post-readout h-only correction. Weak normalized raw-state orth/decorrelation tests cross-state representation geometry. Tenfold lower common LR is a stability hypothesis supported by continuation deterioration; matching control is essential and no causal LR claim is yet supported. Do not add a preservation loss or raw-f weighting change within this factorial without recording a new factor.

## Primary evidence index

- `research/oscillator_r2_20260928/CONSOLIDATED_RESEARCH_CLOSEOUT_20260929_FINAL.md`
- `research/oscillator_r2_20260928/SCIENTIFIC_AUDIT_AND_PROTOCOL.md`
- `research/oscillator_r2_20260928/ensemble_diagnostic/{REPORT.md,FROZEN_CHOICE.json}`
- `research/oscillator_r2_20260928/calibration/{REPORT.md,validation_crossfit.json,exploratory_test.json}`
- `research/oscillator_r2_20260928/frozen_residual/{ROUND_REPORT.md,SCIENTIFIC_RESULT_REVIEW.md,INDEPENDENT_RUN_AUDIT.json}`
- `research/oscillator_r2_20260928/chan64_campaign_completion/CAMPAIGN_REPORT.md`
- `research/oscillator_r2_20260928/eta0_seed_replication/completion_receipts/SEED_FAMILY_SCIENTIFIC_CLOSEOUT.md`
- `research/oscillator_r2_20260928/final_f5_historical_test/TEST_RESULTS.json` (eta0 context row and TRAIN-derived slices)

## Git/workspace cautions and next ownership

Server authoritative working directory has no `.git`; git is absent from its normal PATH. Local task cwd is old dirty branch `codex/qm9s-eta-ef-exploration`, HEAD528f2d4; preserve unrelated edits. `D:\MTO\publication\MTO-1` is a Git plumbing publication directory: local main is unborn; `origin/main` and `origin/codex/oscillator-r2-research-20260928` resolve to69bf39f, while local research branch is3d064bf. Do not assume its empty worktree is a populated clone or overwrite it. This audit read the actual69bf39f commit object and tree. No project AGENTS.md was found in the server tree or local task root.

Local historical final archive: `D:\MTO\archives\oscillator_r2_20260928\round23_final_closeout`. This new handoff is not yet committed/pushed. Root owns decisions; science owns implementation; monitor owns four-hour scheduling/archive. History agent next performs independent prelaunch review without editing training source.
