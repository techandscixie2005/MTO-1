# Scaled native DetaNet baseline: bounded design audit

Date: 2026-09-29. Decision: WAIT for current loss/four-head validation summaries. This note is the only new artifact. No training, job polling, test-label access, configuration changes or implementation occurred.

## Findings

MTO already uses the full DetaNet 128-channel, three-block, l<=3 backbone, with32 trainable Bessel radial functions, radius5 Angstrom and eight attention heads. Recursive comparison found official_detanet/detanet_model and frozen_reference/upstream/vendor/detanet_model identical, excluding bytecode caches. Native DetaNet is therefore a routing/readout and optimization control on a shared backbone, not a backbone replacement.

Verified parameter counts: common backbone1,371,840; historical native E/f1,391,188; original MTO1,552,092; current direct_f MTO1,551,996. Direct MTO has160,808 more parameters than native, mainly molecular routing/coupling/readout. Do not silently widen native to force parameter equality.

Historical native optimized, selected and scheduled on unscaled joint raw E/f MSE, validated every50updates, allowed negative f and used a different stopping/clipping policy. Its validation f R2~.0463 is not a fair estimate of DetaNet's attainable f performance. With LE+Lf denominators sE2=.5378066634062587 and train Var(f)=.002510981243894732, the relative coefficient on f error rises about214.18 times over equal raw E/f coefficients. This is not a measured gradient-ratio claim. Train f_std=.050109692115345626; existing statistics suffice without rescanning labels.

Native readout applies128-to128-to20 MLP to each final atomic scalar S, then sums. S already mixes geometric/equivariant information in three message-passing blocks. However native lacks MTO's global composition context, state-conditioned routing, reference/excited-state CG contractions and global tensor Gram terms. Squared routed tensor sums include cross-atom/cross-channel interactions. Direct MTO keeps routing/context/CG scalar features but loses some tensor Gram information outside hidden h; native pooling removes further global nonlinear readout structure. Softplus AFTER pooling makes the output nonadditive but does not reconstruct those contractions. Effects must be attributed to the readout family jointly. The current original-versus-retained-residual comparison better isolates added scalar correction while preserving tensor information.

## One conditional next experiment

If the current matched screen leaves a material unresolved gap, run ONE from-scratch matched pair:

- MTO-direct: existing direct_f architecture, genuinely initialized from scratch.
- Native-direct: official scalar20-output DetaNet with positive molecular E and direct f.

Both emit E=softplus(u_E+b_E), f=f_std*softplus(u_f+b_f). Transform at molecular/state level, AFTER native atomwise summation. Use train-only per-state mean inverse-softplus offsets and zero final output affine weights/biases so both start at the same E/f means. Newly introduced fixed offsets are buffers; verify actual parameter counts. First-step zero backbone gradients from zero final weights are intentional. No teacher warmup, pretrained weights, tensor loss, f/E product or transformed-label target.

Decisive comparison: native versus MTO direct-f validation SSE at equal examples/updates. This removes old loss scaling and pretrained-adaptation confounds and holds positivity/direct-f form fixed. It tests whether MTO molecular processing helps or hinders learning at this budget. It does not isolate tensor squaring, addressed by the current head screen. Scratch native versus pretrained MTO continuation alone is not a causal architecture comparison.

Controls and bounded budget:

1. Same connectivity split120355/6686/6686, raw printed f, all ten states/masks, radius graph cache and z,pos inputs. No true E input to f. Assert molecule/state alignment. A new loader should restrict labels to train/validation: existing Data eagerly loads all labels including test, so do not reuse it blindly under a strict no-test-access requirement.
2. Paired seed11 initially. Construct one randomly initialized common core and copy it exactly to both arms. Same order hashes, Adam AMSGrad, batch64, decay0, clipping5, FP32, AMP/TF32off, two CPU threads.
3. Exactly100epochs,1881updates per epoch,188100updates per arm. Proposed fixed schedule: lr1e-3 epochs1-60,3e-4 epochs61-85,1e-4 epochs86-100. No validation scheduler/early stopping. This is a controlled screen budget, not an established optimum. High train error or late improvement at the budget limit makes a negative result inconclusive. Do not reuse the continuation20epoch budget or historical million-update stopping interpretation.
4. Evaluate epoch0 and every epoch; choose minimum pooled raw-f validation SSE, earliest tie. Save common-objective best separately, complete curves and full train metrics at selected checkpoints. Report float64 pooled/per-state R2/SSE/MAE/RMSE, E MAE, fixed bright-tail SSE, train/val loss and gradient norms. No clipping search/state filtering.
5. Before launch verify rotation/permutation behavior, edge-cache identity, finite gradients, identical common-core weights and initial mean predictions, masks, checkpoint/resume equivalence, source/data/split/config hashes. Keep all sealed files immutable.

Budget:100epochs is five times a current20epoch arm. Historical MTO145-150sec/epoch suggests4.0-4.2GPU-hours per MTO arm before overhead; native runtime is unmeasured. Report updates/examples AND measured GPU time. Admit only after current results/resource review without preemption; this note authorizes no launch.

A>=.01 validation R2 advantage with positive delta in>=80% of2000 paired molecule bootstrap draws is exploratory evidence favoring that readout family. Reverse advantage supports MTO at its larger capacity. Both poor does not establish a backbone ceiling: low train error with poor val suggests generalization; high train error can reflect optimization, representation or readout. Bootstrap after checkpoint selection is descriptive. E regression is a reported tradeoff, not automatic f veto. No test evaluation. Promising results need matched-seed confirmation and genuinely unseen data.

## Prepare now or wait

WAIT on runner/config preparation. Current four-head results may identify a retained-information correction or a new-head optimization problem and change this pair's value. Root should decide after those summaries whether roughly eight or more additional GPU-hours are justified. Deeper or state-conditioned message passing is a different future backbone intervention; no additional experiment is designed here.

## Sources inspected

Within experiments/qm9s_eta_Ef_20260926: detanet_adapter.py, trainer_detanet.py, dataset.py, model_factory.py, configs/detanet_ef.json, configs/mto_eta0.json, frozen_reference/models_ea.py, frozen_reference/upstream/models.py, official/vendored detanet_model sources; reports/detanet_interface_checks.json, detanet_confirmation.json, detanet_protocol.md, execution_snapshot_detanet.json, detanet_implementation_commit.txt. Historical implementation commit recorded528f2d4f8e373c99ae4cbb39baa5bd5d3cfd326c.

Research sources: SCIENTIFIC_AUDIT_AND_PROTOCOL.md, TOPOLOGY_BASELINE_FEASIBILITY.md, architecture/ARCHITECTURE_PROTOCOL.md, architecture/model.py and architecture/stats.json. Source snapshots/hashes remain authoritative; do not infer a current remote Git commit. No current run histories/statuses inspected.
### Storage access clarification
The eager Data loading observation concerns storage access discipline only. Loading test arrays into memory is not evidence that test labels influenced gradients, checkpoint selection or scientific decisions. No training/selection leakage finding follows from this observation, and no sealed current study needs alteration solely because of it. Future implementations should restrict actual label consumption to authorized train/validation indices and can additionally avoid eager test-array loading for clearer access auditing.