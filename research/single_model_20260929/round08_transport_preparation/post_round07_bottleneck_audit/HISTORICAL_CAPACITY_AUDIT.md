# Historical capacity and training audit after Round07

## Verdict

The bounded history supports a new **explicit neighbor-tensor transport versus local-tensor residual** question. It does not establish that missing transport causes the errors. No matched transport or backbone-depth experiment was found in the inspected sources and completed reports. Earlier width, learning-rate and batch-size trials do exist, so those must not be described as wholly unexplored.

The strongest competing explanation remains generalization or optimization behavior: TRAIN fitting continues while validation raw-f worsens, and declared seed11 original controls vary materially. The proposed transport contrast is useful because the missing direct tensor edge input is an exact source fact and its local control tests a simpler residual alternative. Neither fact guarantees a gain. A regularization-only study remains scientifically plausible; this audit does not rank all possible future hyperparameters or recommend a sweep.

This is source/configuration/existing-aggregate review only. No arrays, checkpoint tensors, models, inference, new target values, fits or TEST scoring were used. All closed artifacts remain unchanged. The new evidence receipt contains23 configurations and92 exact source/input hashes: `HISTORICAL_RECIPE_EVIDENCE.json`, SHA840e5c55eac1806f65187bc3cd55ec2c5d7bd6d3d4f12856308cc9f2d8925b81. `read_history_metadata.py` records the bounded read set. This is not a repository-wide proof of absent experiments.

## 1. Completed full-E/A controls existed before the current campaign

Source root: `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925`. The exact configurations, histories and completion-marker hashes are in the evidence receipt. `models_ea.py` uses normalized energy MSE plus **full tensor Frobenius MSE**, LE+LA. This is different from the current trace-only LE+Ls. Model selection and the adaptive scheduler use joint validation LE+LA. The table reports that original objective, **not raw-f R²**.

| MTO trial | Changed setting relative to historical reference | Completed / selected epoch | Best validation LE+LA | Updates | Parameters |
|---|---|---:|---:|---:|---:|
| reference | channels16, LR.001, batch64 |248 /45|.5233464623|466488|1,552,092|
| channels32 | MTO channel count16→32 |244 /92|.5318217367|458964|1,783,820|
| lr3e4 | initial LR.0003 |226 /23|.5378911514|425106|1,552,092|
| lr3e3 | initial LR.003 |227 /75|.5399122191|426987|1,552,092|
| batch128 | batch64→128 |236 /33|.5304371999|222076|1,552,092|
| seed23 | initialization/order seed23 |194 recorded /64|.5468941284|364914|not a completed confirmation|

The first five rows have FIT_COMPLETE records; seed23 has FAILED and no FIT_COMPLETE. Its historical CUDA failure must not be counted as a negative completed seed experiment. The DetaNet reference also completed259 epochs, joint best.576526582 at107; its different architecture is not an MTO width control.

Common declared settings: query32/router128/head128, AMSGrad, weight decay0, clip5, FP32/noAMP/noTF32, threads2. ReduceLROnPlateau uses factor.5, patience50, relative threshold1e-4 and minimum LR1e-6. Stopping requires the original minimum200/patience150/reduction conditions, so realized epochs and learning-rate histories differ. Batch128 changes update count; altered layer shapes do not preserve a bitwise common initial parameter realization. These are useful historical recipe comparisons within LE+LA, but not matched current v2 native-f experiments.

`reports/selection_before_test.json` explicitly records that historical TEST was opened before all fits completed. We read only its selection/protocol metadata. We did not extract a new raw-f result from this campaign or reinterpret joint loss as one.

## 2. Channel64 also changed trace weighting and tensor supervision

Source root: `/home/inspur/MTO-1/experiments/qm9s_chan64_20260928`. All four configs set `trace_weight_E2:true`. `objective.py`, SHA6de50ddaa5edc1f7fc232a67ad212827cfbf3430eef48537a502e463e6a5c131, weights trace squared error by true E² divided by TRAIN mean E². This applies to **G1 as well as G3**; a description that mentions it only for G3 is incomplete.

| Run | Architecture / objective distinction | Completion and selection | Recorded validation raw-f evidence |
|---|---|---|---:|
| G1 | Original PSD,64 MTO channels, E² trace plus LE and Q loss |227, selected75|native .40525459|
| G2 | Original PSD,64 channels, E² trace only; E head frozen/unsupervised |intentional stop after214, selected41|oracle-E .38078050; ineligible as native predictor|
| G3 | Vector outer-product tensor,64 channels, E² trace plus LE and Q |326, selected174|native .36810069|
| G4 | Vector outer product,64 channels, E² trace only; no learned E supervision |223, selected71|oracle-E .34833500; ineligible as native predictor|

The native G1 figure is .40525454066 in the saved CPU audit; the report rounds it. Source-backed completion evidence is `research/oscillator_r2_20260928/chan64_campaign_completion/{CAMPAIGN_REPORT.md,CAMPAIGN_AUDIT.json}`. Selection uses each original objective, with adaptive scheduling and unequal stopping. G1 has3,451,500 parameters, G3 has3,393,195; the backbone remains1,371,840 in both. None is a channel64-only comparison with current v2 LE+Ls. The F5 ensemble is irrelevant to the single-model eligibility question.

Earlier channel16 `qm9s_pouter_trace_20260927` and `qm9s_pouter_trace_wE_20260928` configs establish that original/vector and trace/Q/E supervision variants existed. The fixed marker inventory finds no completion marker for the former four, and completed weighted trace-only G2/G4 in the latter. Do not convert incomplete historical logs or unsupervised/oracle-energy diagnostics into clean native comparisons. Their exact observed statuses and latest aggregates are preserved without a new process or numerical audit.

## 3. Which capacities and regularizers actually changed?

The shared `upstream/models.py` in the first campaign and eta-Ef frozen reference is byte-identical, SHAa8d7bf8a8d89839f687a773f59f02b2a5d75abd6f47a59406e9eb3d1a40f4771. The core configuration is128 features, three interaction blocks, l≤3,32 trainable Bessel features, eight heads and configured5Å radius, dropout0. Current production supplies the pinned cached edge list; this audit did not measure its distances or graph diameter. MTO channels16/32/64 change the projection, router output, CG/readout dimensions; they **do not increase the DetaNet core width or block count**. A source option for depth, dropout or normalization is not an executed ablation.

All23 inspected early configurations set weight decay0. No clean backbone-depth/width, dropout or weight-decay comparison was found in this bounded ledger. Later Round01/05 raw-M decorrelation is a distinct explicit regularizer, already tested and not a reason to repeat it unchanged. The history does not justify a claim that regularization cannot help; it also provides no validated regularization coefficient to recommend.

## 4. Schedules, seeds and objective changes are not interchangeable

- **Old eta0:** `/experiments/qm9s_eta_Ef_20260926` uses LE+Ls+eta LQ; eta0 retains energy supervision. It completed236, selected33 on its joint objective, with the original plateau/stopping recipe. Native old-validation R².405294118 is the historical anchor. The calibrated .418119 full-validation refit and .416658 OOF evidence are old-split calibration evidence, not v2 performance. In the later G-series, “eta0” instead means trace-only without energy supervision; names alone do not identify the same loss.
- **Seed23/37 replication:** `research/oscillator_r2_20260928/eta0_seed_replication` ran original LE+Ls for100 epochs with the legacy scheduler. Seeds change both initialization and minibatch order. Best epochs53/44 give .368708895/.394455334; seed11's best33 lies within100 but its source ran236. Selected and final TRAIN R² increase (.778586→.884062 and .725204→.904943), while VAL falls (.368709→.336711 and .394455→.352688). These are saved completed aggregates; no TRAIN arrays were recomputed here. This supports a generalization question, not a model-capacity ceiling or a fully matched three-seed budget claim.
- **Direct-f scratch:** `scratch_readout_comparison` changed the PSD head to a direct-f head, used LE+Lf, altered zero-final/fixed-energy initialization, and ran100 epochs with LR.001 through60, .0003 through85, .0001 through100. MTO .373492948 versus native .343781976 does not isolate an LR schedule or original PSD expressivity.
- **Recent objective/readout trials:** full-model warm-start LE+Ls/E²/raw-f continuations at1e-4 and Round01's controlled1e-5 factorial did not improve the anchor. Round02's frozen-F objective pair had only .000114/.000093 native gains. Round04 f-only fixed hinges underperformed affine. Round05 fresh F/decor failed; Round06 fresh original-PSD raw-f beat its own control but failed the retained-reference gate; Round07 congruence failed against scalar and retained. These results rule out unchanged repeats as the next study, not every other optimizer or representation change.

## 5. Repeated control variability constrains any next claim

The current retained checkpoint is Round05 original seed/order11 epoch45, validation raw-f R².44716940136585204. Same declared original recipes in Round06 and07 selected .4047975673 and .4091040813. The original model, initial base, data order, masks/statistics and declared optimizer schedule match, but diagnostics/autograd paths and concurrent workloads differ. The published `round06_objective_preparation/completion/CONTROL_REPEAT_CONTEXT.md` records these differences. No cause was experimentally isolated, and these are not independent-seed confirmation.

A new neighbor/local/original pilot must use one common implementation of all unintended paths and report selected versus aligned final epoch separately. Require the candidate to exceed both contemporaneous controls **and** retained.447169401 by the proposed .003 before allocating confirmation. That screen is not a significance test or robustness claim. Independent fresh seeds with within-seed controls are necessary for a broader claim, with individual one-checkpoint results and no prediction averaging. Large state/tail/energy regressions still require a decision; a pooled threshold does not certify their acceptability.

## 6. Independent mathematical challenge to the proposed path

The vendor `Message.forward(S,rbf,sh,index)` has no T argument. `Update.forward` sums into `j=index[1]`; `Edge_Attention` uses query S_i, key/value S_j. Use these operational indices, without treating the naming as a defect. Source T can still affect later edges through local invariant contractions into S. Global molecular pooling and CG already produce cross-atom bilinear terms, so the whole predictor does not lack cross-atom interaction or coherent squares.

At fixed intermediate S/radial/receiver-T, direct neighbor-T addition creates a dependency absent from the original edge function. This is a computational distinction, **not** an exhibited pair of physically realizable geometries that the original model cannot distinguish. Nor does finite128→16 compression alone prove that more channels would help.

The proposed same-global-irrep-basis transport is O(3)-compatible: scalar gates and same-irrep sums commute with the representation. Per-channel theta must be shared over magnetic components. With two blocks, three irreps and128 channels,768 parameters is correct. Theta0 nests the old algebra; d(delta)/dtheta is the gated neighbor/local feature mean. Particular zero or symmetric inputs can still have zero gradients, so liveness needs the proposed bounded future fixture, not an assertion of universal nonzero gradients.

The local residual has identical gate inputs, parameter count and insertion point but a different function class and activation distribution. Degree averaging plus tanh bounds weights, **not** the residual relative to receiver T or original message. Fixed diagnostics should report that ratio without adaptive rescaling. An apparent gain would support this entire transport recipe over its chosen local comparator, not isolated angular expressivity, electronic coherence, TDDFT physics or a universal capacity limit.

The proposal is defensible on those terms. It is one falsifiable upstream question; no implementation or execution follows from this audit. Root selects the next protocol and controls separately.
