# Matched scratch readout pair: independent scientific closeout

Recorded at 2026-09-29T08:08:04.129273+08:00. **Completed-artifact and paired-result verification PASS.** Neither selected standalone nor their prescribed equal mean replaces F5 on pooled raw-f validation R². The separately frozen F7_scratch analysis remains gated on its implementation review and is not evaluated in this note. No training extension, new architecture, weight search or test access follows.

## Primary result

| Predictor | Selection | Raw-f validation SSE | R² |
|---|---|---:|---:|
| Native direct positive readout | raw-f epoch 21 | 103.949028 | 0.343781976 |
| MTO direct positive readout | raw-f epoch 16 | 99.242625 | 0.373492948 |
| Prescribed equal scratch mean | the two selected arrays | 94.547365 | 0.403133570 |
| Archived eta0 | historical legacy selector | 94.205121 | 0.405294118 |
| Original equal3 | fixed loss ensemble | 87.215074 | 0.449421463 |
| Frozen F5 | fixed five components | 81.151578 | 0.487699605 |

The primary native-minus-MTO ΔR² is −0.029710972, with paired molecule interval [−0.052854670, −0.007184501] and connectivity-group interval [−0.054924942, −0.006530601]; positive fractions 0.004 and 0.0025. I reproduced the frozen 2,000 draws with seed 20260928 and sorted connectivity groups. This supports MTO's readout family within this one matched fresh-core/order/normalized-objective experiment. It does not establish a universal architecture ranking: the routing, nonlinear readout, gradient paths and parameter counts differ (native 1,391,188; MTO 1,551,986; common core 1,371,840). This is a shared-backbone readout comparison, not a replacement of the backbone. Equal epochs/examples are not equal compute; matched fresh core does not make unequal heads identical initialization functions.

MTO improves nine of ten state SSEs versus native (S3 is the exception), with its largest gain at S8 (1.681259 SSE). Native loses both below-q90 (+1.900834 SSE) and bright q90 (+2.805570), and q99 alone loses 4.361674. Native improves 3,038 molecules and worsens 3,648 relative to MTO. Case14562 contributes 0.694661 of native's 4.706403 excess SSE; the largest negative group contributes 0.886196. The comparison is not entirely a single-case effect, but these are descriptive differences after validation checkpoint selection, not seed-replicated confirmation.

## Training and selection behavior

MTO raw-f and joint selectors differ: epoch16 R² 0.373492948 versus epoch41 R² 0.368982831. The joint checkpoint lowers f MAE (0.018380→0.017728) and E MAE (0.118312→0.107375), but increases raw-f SSE by 0.714431. This illustrates the target-metric mismatch; it does not explain the whole gap to F5.

MTO full-TRAIN R² rises from 0.517471 at raw-f selection to 0.744426 at joint selection and 0.905996 at epoch100, while final validation R² falls to 0.276909367 (SSE 114.542066). From raw-f selected to final, validation SSE worsens 15.299442: below-q90 adds 16.229366 while bright q90 improves 0.929924. Case14562 worsens from 3.295976 to 8.983276 SSE. Final f MAE and E MAE still improve. Native shows the same broad fit/generalization split (selected/final TRAIN 0.584646/0.914208; validation 0.343782/0.227522). Neither final fit supports a backbone-capacity ceiling. The corrected positive, scaled native baseline is much more informative than the old unscaled/negative-output baseline, yet its outcome constrains this fixed schedule/readout/objective combination only.

## Fixed mean and geometry context

The equal scratch mean improves over either standalone, showing some complementary errors, but remains below eta0, old3 and F5. Its ΔR² versus F5 is −0.084566035; a separately prescribed F5 supplemental calculation gives group interval [−0.106463874, −0.066466773] (seed20260929), positive fraction zero. All ten states worsen versus F5. Below-q90 SSE rises 5.438169 and q90 bright rises 7.957618; q99 rises 3.119601. 2,322 molecules improve and 4,364 worsen.

Mean case14562 SSE is 3.631563 versus F5 2.622175 (loss1.009388, the largest molecule/group loss); the six lowest-q2 cases total 6.361208 versus 4.141298. Native/MTO six-case totals are 6.566389/6.585373, respectively, so a lower error on case14562 alone is not a uniform geometry gain. The one-case and five-case bins remain tiny; no geometry-causal, label-error or exclusion claim follows. All 6,686 molecules stay in every headline score. Independent recomputation matches the reviewed geometry report's full/state/q2/case/six-case SSE for both arms and their mean.

## Integrity and scope

Reusing the accepted native proof, I directly verified MTO's genuine 100-epoch/188,100-step completion, 58 source and ten code hashes, all100 native-matched order hashes, fixed LR schedule, copied initial core tensors, exact selected minimum-history epochs, versioned checkpoint/prediction aliases, final/last model tensors and unchanged GPU2 ECC. Raw-selected epoch16 checkpoint is 911ee4260fc4dc70ae627e057c67b530f5d17e7df903a811682d8a5ab10f66c3. Both completed terminals precede paired analyses.

Saved NPZ files omit mask_f_true; exact indices/IDs/raw FP64 f and E truth agree with pinned authoritative all-valid reference masks. No validity was inferred from predictions. Small completion-time replay versus saved-array differences are within declared FP32 tolerances; headline results use the saved selected arrays. That replay occurred inside authorized training completion, not this audit. Full-TRAIN figures remain completion aggregates; this audit checked arithmetic/provenance without new TRAIN inference. Temporary alias leftovers are nonauthoritative: final aliases match selected versioned artifacts and terminal hashes.

Evidence: MTO_INDEPENDENT_VERIFICATION.json; accepted NATIVE_INDEPENDENT_VERIFICATION.json; PAIRED_INDEPENDENT_VERIFICATION.json; GEOMETRY_INDEPENDENT_VERIFICATION.json; unchanged RESULTS.json and reviewed postrun_geometry/SCRATCH_RESULTS.json. Companion provenance pins exact bytes. Independent scripts perform saved-artifact calculations only. Arrays/checkpoints remain server-only; no test was opened. The reused validation and one seed make all intervals conditional/descriptive. Root's final candidate freeze and separately reviewed historical-test policy remain in force.
