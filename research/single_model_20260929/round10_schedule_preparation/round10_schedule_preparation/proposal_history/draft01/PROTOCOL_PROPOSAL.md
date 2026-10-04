# Round10 proposal: one predetermined learning-rate reduction

**Scope: documents and existing source/aggregate evidence only.** Round09 is closed and published at `6e23b9d9c7eeb479b478ebb69ec061024de62455`. This proposal permits no implementation, model construction, numerical test, target/prediction-array decode, inference, fit, new split or TEST access. Independent proposal review and a separate root preparation decision are required next. The directory name does not grant preparation authority.

## 1. Question and one fixed contrast

Does a single predetermined reduction in learning rate improve validation raw-f accuracy within the same 60-epoch budget, using the unchanged original PSD MTO and LE+Ls?

| Arm | Epochs 1–30 | Epochs 31–60 | Other changes |
|---|---:|---:|---|
| `fixed_lr` | .001 | .001 | none |
| `step_lr` | .001 | .0003 | none |

The candidate retains the inherited early learning rate and gives the second half of the fixed budget smaller updates. Epoch31 is the budget midpoint, not a boundary selected by inspecting which past checkpoint won. The .0003 value has historical use but is not validated as this schedule. These are single, unvalidated choices fixed before this future experiment; the proposal was developed after seeing prior source and reused-validation evidence and is not independent of that history. No alternate boundary, factor, warmup, cosine curve, plateau trigger, restart, LR sweep or extension belongs to this study.

This tests the whole two-phase recipe. Equal update count and roughly equal compute do not equalize cumulative learning rate: the sum of the 60 epoch rates is .039 for the candidate versus .060 for control (mean .00065 versus .001). A gain would not isolate an optimal boundary, a pure annealing mechanism, or an advantage over constant .0003/.00065. No extra control or search is silently added.

## 2. Existing evidence and non-repetition

The bounded historical audit inspected 23 configurations and 92 source/input pins. It found no matched original-PSD, LE+Ls, zero-decay, fresh-v2, equal60 comparison of this single reduction against fixed .001. That is a statement about the audited records, not repository-wide absence. Learning-rate and schedule experiments themselves are not new:

| Prior evidence | Actual recipe/result | Why it does not answer this question |
|---|---|---|
| Historical full-E/A reference and LR variants | LE+full-tensor LA; initial LR .001/.0003/.003; best joint validation losses .523346/.537891/.539912; completed 248/226/227 epochs | Different loss, split/selection, adaptive stopping and unequal realized budgets; these are not current native raw-f R² results |
| Historical plateau policy | ReduceLROnPlateau factor .5, patience50, relative threshold1e-4, minimum LR1e-6; driven by original validation objective | A feedback policy with different realized transitions, not a fixed halfway reduction |
| Old eta0 and seed23/37 | Original LE+Ls with legacy plateau; seed11 ran236 and selected33; seed23/37 ran100 and selected53/44 at .368709/.394455 | No contemporary fixed-versus-this-schedule control; different split, seeds and stopping history |
| Direct-f scratch comparison | Changed PSD readout to direct scalar-f, LE+Lf and altered energy/f initialization;100 epochs with .001 through60, .0003 through85, .0001 through100; MTO .373493 | Head, loss, initialization and budget changed; the first60 had no LR drop |
| Current original v2 controls, Rounds05–09 | Fixed .001 through60; selected R² .447169, .404798, .409104, .417636 and .415284 | Relevant fixed-rate evidence, but no scheduled counterpart and substantial observed run-to-run variation |

Source-backed details are in `../post_round07_bottleneck_audit/HISTORICAL_CAPACITY_AUDIT.md`, `HISTORICAL_RECIPE_EVIDENCE.json` and the published round reports referenced by REFERENCE_MANIFEST.json. Earlier channels32/batch128 trials also completed under LE+LA/plateau; the seed23 full-E/A run failed and is not a completed negative confirmation.

In Round09 zero_decay, TRAIN trajectory raw-f MSE fell from .000767 at selected epoch39 to .000479 at60 while validation R² fell from .415284 to .407024. These TRAIN values are pre-update minibatch aggregates, not fixed-checkpoint evaluations. Late validation decline is compatible with several explanations; it does not establish optimization failure, a learning-rate cause, or a proven generalization gap. Smaller later steps could help, do nothing, or inhibit useful progress.

Round09 coupled decay failed despite substantial parameter/radial/offset changes. This does not justify tuning decay or asserting a radial-parameter mechanism. The present candidate sets decay to zero for both arms and changes only the LR schedule. Repeated-control variability remains unresolved; original initial tensors/order matching does not promise bitwise CUDA trajectories. Use contemporary identical source and diagnostics in both arms, and retain the stronger-reference gate.

## 3. Exact epoch/update convention and optimizer behavior

Epoch0 is evaluation of the fresh initialization and performs no optimizer update. Training epochs are one-based e=1,...,60. Each contains 1,881 minibatches, including the final shorter batch, giving 112,860 updates. For one-based optimizer update u:

    e(u) = floor((u-1)/1881) + 1
    fixed_lr(e) = .001                         for 1 <= e <= 60
    step_lr(e)  = .001                         for 1 <= e <= 30
                 .0003                        for 31 <= e <= 60

Thus updates1–56,430 use .001 in both arms; updates56,431–112,860 use .001/control and .0003/candidate. Rate is constant within an epoch. Freeze the explicit 60-value table for each arm in PROPOSAL_SETTINGS.json and later bind its exact contents in checkpoints/manifest. No validation quantity enters this table.

Before the first minibatch of each epoch, assign the absolute prescribed rate to the single optimizer group and assert it. Do not multiply the saved rate cumulatively, infer a phase from validation, call a metric-driven scheduler, or count a scheduler step as an optimizer update. No final extra update is performed. Log the rate actually used for that completed epoch, schedule phase, global update range and optimizer batch count. No within-epoch scheduler step exists.

Use the same pinned torch Adam with AMSGrad, betas(.9,.999), eps1e-8, weight_decay0, grad clip5, foreach=None/fused=None, capturable/differentiable/maximize=false. Keep the exact task-backward → inherited clip_grad_norm_ → detached common diagnostics → optimizer.step ordering. Adam first/second/AMSGrad-max moments and per-parameter step counters continue across the rate transition; no reset, rescaling, optimizer reconstruction or decay term is added. `grad=None` retains inherited skip behavior. Lowering LR changes the final adaptive parameter-step multiplier, not the task loss, gradient clip or moment recurrence; subsequent gradients naturally diverge as parameters diverge.

Both arms contain the identical ordered135 original trainable tensors (1,552,092 values), including biases, normalization, embeddings, offsets and radial parameters. Existing roster SHA `290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943` is the reference. Exclude only the existing11 frozen right-F/transport tensors and buffers. Both use original mode: right-F disabled, two transport theta tensors frozen at zero, original PSD decoder unchanged. Do not add parameter groups, exemptions, weight decay, dropout, clipping or diagnostic-graph differences between arms. Later preparation must reverify the full ordered roster against a fresh unwrapped original model, without inheriting trained weights.

## 4. Resume and checkpoint semantics

Preserve atomic complete-epoch `last.pt` as the sole committed resume point. It contains model, Adam/AMSGrad, all RNG/order-generator state, completed epoch/update count, history, selection references, full config/statistics/roster and exact source/split/schedule bindings. `best.pt` remains a selected snapshot, not the current committed transaction. One checkpoint serves geometry-only deployment; schedule/optimizer metadata is provenance, not a new inference dependency.

Record `lr_used_for_completed_epoch`, `schedule_phase_used` and `next_epoch_lr` alongside the full table. Epoch0 uses the initial optimizer rate .001 but records no used training LR/phase; next_epoch_lr=.001. At epoch60, next_epoch_lr is null. The optimizer's stored LR at a committed epoch must equal that epoch's applied rate (except epoch0's initial rate).

At committed epoch30 the candidate optimizer LR is still .001. Restore the complete Adam and RNG/order state first; then, before epoch31's first update, assign .0003 from the absolute table. An interruption within epoch31 replays from committed30 with the same transition and order. At later resume points, assign the table value for completed_epoch+1 and verify the saved last-used rate, rather than applying the drop twice. Resume from59 uses the epoch60 rate; completion at60 refuses relaunch. No skipped/duplicate optimizer or order advances are permitted.

Preserve failed attempts and original identities. An app interruption is not a training failure. Any numerical/resource failure is recorded and stops the affected run; no LR fallback, device reassignment, blind restart or completed-stage repeat is automatic. Later implementation must coherently replace inherited fixed-LR assertions in training, preflight, recovery, metadata inspectors and terminal audit; it must not simply remove LR checks.

## 5. Data, objective, initialization and budget

Use sealed v2: TRAIN120,355 / validation6,686 / TEST6,686. Split manifest `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent verification `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`. Disjointness is under conservative audited identity rules, not absolute chemical-identity proof or external fresh data. TEST remains sealed and contains5,989 historicalTRAIN/361 historicalvalidation/336 historicalTEST rows. No split change or target exclusion.

Use the existing newTRAIN-only statistics, SHA `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`: sE²=.5376833706691944, sA²=.13245475393297818, n_ref18 and exact frozen state means. Preserve original base_loss:

    LE = mean((Epred-Etarget)^2 over valid E) / sE²
    Ls = mean((trace(Apred)-trace(Atarget))^2 over valid E AND valid A) / (3*sA²)
    loss = LE + Ls

Inherited linear trace extraction before masking stays unchanged; squared-error arithmetic is on valid indices. All frozen benchmark masks are true; reject a changed contract rather than drop rows. No raw-f, full-Q, E²-weighted or decorrelation objective. Geometry/readers/model are unchanged: core128/three blocks, MTO16/query32/router128/head128, dropout0, FP32/noAMP/noTF32/two threads. No teacher energy, clamp, calibration or averaging. Native raw-f evaluation is FP64 `2*E_eV*trace(A)/(3*27.211386245988)` against printed f, preserving zeros and all66,860 validation labels. Softplus energy and PSD A maintain nonnegative f; reject invalid/nonfinite outputs without repair.

Both arms start independently from the same fresh seed11/base/full tensor realization and use order_seed11, all60 pinned TRAIN permutations. Base reference hash `231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2`; unchanged schema full hash `f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3`. Do not fork a trained control at30, use old weights, or reuse disposable technical states. Both full60 runs are needed for a contemporary source-matched comparison. No early stopping, warmup or extra epoch. There is no requirement that separate CUDA trajectories remain bitwise equal during their shared-rate prefix.

Recent Round09 original fits took about3.09hours each. Plan roughly6.2 GPU-hours for the pair, or roughly3.1hours wall time on two independently admitted devices if available, plus separately bounded future engineering checks. The schedule adds negligible state/compute relative to the unchanged model; actual timing and peak memory remain measured quantities. No devices are reserved or admitted now.

## 6. Selection, promotion and reporting

For each arm, select the earliest minimum pooled validation native-f SSE among epochs0–60. Report that checkpoint and fixed60; neither policy is changed after results. The candidate is eligible only for a later root seed-confirmation decision if selected R² is at least .003 above BOTH contemporaneous selected `fixed_lr` and retained Round05 reference `.44716940136585204`. A weaker rerun alone cannot establish progress. This is an allocation screen, not statistical significance or proof of generalization.

A pass requires full terminal/source/optimizer/order/access review, honest state/tail/energy results and archived closeout before root decides any confirmation work. Any later seeds must be new matched control/candidate pairs with the same frozen schedule and budget, reported separately; no best-seed selection or prediction averaging. No seed is authorized now, and a pass launches none automatically. A failure closes this exact schedule/budget without alternative rate/boundary, extra epoch or TEST scoring. A better fixed-rate control may be retained descriptively by root without a schedule claim or automatic allocation.

Retained reference: Round05 control45 geometry_best.pt, SHA `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`; still below .60 and without current TEST/independent-seed confirmation. Use its matching loader. No results from incompatible splits are pooled.

Report all ten states, pooled MAE/RMSE/R²/counts, energy MAE/RMSE, true-tail thresholds q90=.0549/q99=.2406, full true/predicted-q99 bins, error concentration and selected/fixed60 checkpoint hashes. False-bright membership is model dependent. Use one saved-output analysis, no new model inference; paired identity-component bootstrap2000 draws/seed20260930/recomputed pooledSST, all states/molecules within each sampled group together. Draw molecule counts vary; intervals condition on reused validation/selection and exclude training-seed uncertainty. No paired interval against an aggregate-only retained reference.

Plot all60 applied LRs with aligned validation and TRAIN curves, checkpoint boundary30/31, task/parameter/radial/offset diagnostics and optimizer step counts. Retain common detached zero-decay diagnostics in both arms. TRAIN values remain pre-update minibatch aggregates, not fixed-checkpoint TRAIN evaluations. Do not use diagnostics to adjust LR, clipping or any setting. No physical phase, density, sum-rule or QC-mechanism claim follows from a schedule change.

## 7. Future preparation boundary — not executed

Only after a separate root preparation decision: implement one shared runner, explicit LR tables and full fail-closed production authority. Freeze source/roster/initialization/order/tolerances before numerical work; history independently reviews source and actual results. Final acceptance, D-first publication and exact root production authorization precede either60-epoch run. Publication alone never grants fitting rights.

Propose bounded checks, not tests already performed:

1. Metadata/synthetic checks cover all60 rates and exact updates56,430/56,431, epoch0/30/31/60, forbidden positions, no duplicate transition, schedule/config hash rejection, and checkpoint LR fields before/after resume. Use synthetic optimizer state to verify persistent Adam/AMSGrad moments/step counters and the intended LR multiplier, not reset/reconstruction. Fix exact toy-call counts and manual FP64 tolerance before this later stage; no numerical tuning now.
2. Fresh CPU constructors verify original/paired roster and initial hashes, unchanged loss/masks/symmetry and strict one-checkpoint geometry export/access. No historical model or dataset is needed for those synthetic checks.
3. Proposed full-model technical cap: first128 fixed TRAIN rows, two64 batches, three executed updates per arm/six total. In an explicitly labeled boundary fixture, update1 uses the table's epoch30 LR, update2 uses epoch31, then restore afterupdate1 and replay update2. Actual Adam counters are1/2, not a claim of30 completed training epochs; synthetic schedule-position labels only exercise the boundary. All fixture states are discarded. Exact RNG/order replay and inherited model/Adam atol2e-6/rtol1e-5, loss atol1e-6/rtol1e-5; no outcome-dependent relaxation. No validation/TEST fixture or additional update. Full future production boundary progress is audited from actual completed histories.
4. Registration-before-compute, CPU/GPU prebinding, fresh health/UUID/shared locks, owned identities, immutable attempts/completion, partition-safe readers and source/review/publication/auth bindings remain required. No resource choice is authorized by this document.

The outcome can support or reject this fixed two-phase learning-rate recipe at this budget. It cannot prove that earlier degradation was caused by excessive LR, isolate the cumulative-rate effect, establish a schedule optimum, reject all schedules on a null, or supply new-test/seed confirmation on a gain.
