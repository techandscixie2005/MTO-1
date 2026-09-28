# Seed family and fixed-seven independent scientific closeout

Recorded at 2026-09-29T07:40:11.383696+08:00. Saved artifacts only; no model inference, test access, active scratch outcomes, training extension or mixture search.

## Decision

Retain F5 as the provisional validation candidate from these completed recipes. The extra two seeds do not improve the prescribed seven-way mean: F7_seed R² 0.486690951 versus F5 0.487699605, at seven versus five neural forwards. This is a small observed difference with uncertainty spanning zero, not proof that seed diversity cannot help. No seed expansion or test query is justified by this closeout alone. Final candidate selection remains with root after the remaining approved studies; any test comparison requires a separately frozen and reviewed recipe. The historical test has been reused and cannot be presented as fresh confirmation.

## Integrity and recomputation

Independent CPU checks verified both genuine 100-epoch/188,100-update terminals, all 100 prescribed order hashes per seed, exact reviewed source/code hashes, selected checkpoint byte hashes and minimum-history selection. All saved initial/selected/final arrays align with the fixed 6,686 validation IDs/indices and raw FP64 truth. The authoritative reference masks are all valid; the absent seed NPZ mask field was handled by the separately reviewed exact-path/hash/ID/truth compatibility adapter. Its original schema failure and sealed source remain preserved. No mask was inferred from prediction values.

For each seed, the legacy and raw-f selectors choose the same epoch: 53 for seed23 and 44 for seed37. All 178 selected model tensors/buffers are bitwise identical between the two aliases, as are their saved predictions. The serialized checkpoint files differ because selection-name and metric metadata differ; this is not floating-point rounding. Other checkpoint metadata agrees. The primary legacy-selected three-seed definition and mixed-selection secondary remain distinct despite identical results here; the latter supplies no independent gain or evidence.

| Predictor | Validation SSE | Pooled raw-f R² |
|---|---:|---:|
| Archived eta0 seed11 | 94.205121 | 0.405294118 |
| Seed23 legacy, epoch 53 | 100.000448 | 0.368708895 |
| Seed37 legacy, epoch 44 | 95.922052 | 0.394455334 |
| Primary seed3 (both selectors coincide) | 86.655328 | 0.452955075 |
| Original equal3 loss ensemble | 87.215074 | 0.449421463 |
| Frozen F5 | 81.151578 | 0.487699605 |
| F7_seed_legacy | 81.311355 | 0.486690951 |

Seed23 selected/final full-train R² is 0.778586/0.884062; validation is 0.368709/0.336711. Seed37 is 0.725204/0.904943 on train and 0.394455/0.352688 on validation. Final validation SSE rises to 105.069155/102.538225 while MAE and energy error improve. Thus these runs increasingly fit training data and worsen the target squared-error metric; they do not establish a backbone capacity ceiling. Full-train values are completion aggregates with arithmetic/provenance checks, not independently recomputed train prediction arrays. Old seed11 used a historical early-stopped run, whereas the new runs used fixed 100-epoch budgets: this is not a matched-budget estimate of seed variance.

## Uncertainty and distribution

I independently reconstructed all fixed means and reproduced every paired interval for seed3 versus eta0/equal3 and F7 versus F5/seed3. Seed3 uses 2,000 draws with seed 20260928 and first-encounter group order; F7 uses seed 20260929 and sorted groups, matching their respective frozen rules. All ten states remain together; pooled SST is recomputed for each draw across 6,671 connectivity groups. These intervals are descriptive conditional on repeated validation selection, not selection-adjusted confidence or fresh replication evidence.

- Seed3 minus equal3: ΔR² +0.003533611; group 95% interval [−0.037479786, +0.029682444], positive fraction 0.618 (molecule interval [−0.039888130, +0.029298183]). Bright q90 SSE improves 2.074819 but below-q90 worsens 1.515073; S7 loses 2.248184 SSE. Case14562 alone worsens 2.660735 SSE. The net gain 0.559746 is uncertain; 3,750 molecules improve and 2,936 worsen.
- F7 minus F5: ΔR² −0.001008654; group interval [−0.020518744, +0.010437931], positive fraction 0.488 (molecule [−0.020300527, +0.010452894]). Eight states improve; S7 loses 1.191074 SSE and S9 loses 0.008956. q90 bright and q99 SSE improve 0.974135 and 0.508864, respectively; below-q90 worsens 1.133912. Across molecules, gross gains 3.535850 and losses 3.695627 yield net SSE deterioration 0.159777; 3,795 improve and 2,891 worsen. Largest positive molecule/group ID52479 gains 0.105323; largest negative molecule/group ID14562 loses 1.320024 (SSE 2.622175→3.942199). The six lowest-q2 cases worsen from 4.141298 to 5.496182 SSE. The higher three fixed q2 bins improve; the lower three worsen. Negative net gain has no positive gain-share headline.
- F7 minus seed3: ΔR² +0.033735876; group interval [+0.014659340, +0.067883549], positive fraction 1.0. All ten states improve and total SSE falls 5.343973, but q99 worsens 0.629239. ID14562 contributes 2.315077 (43.3%) of this net gain; concentration matters even when a whole-benchmark improvement is valid.

The near-linear stratum has only one case and the next stratum five. Their errors are useful diagnostics, not evidence geometry causes error or grounds to exclude/downweight valid cases. Absolute SSE, not just shares, supports these statements. F7 versus F5 changes both the added predictors and the equal weights on the original five; it does not isolate a causal seed effect. Forward counts are not calibrated latency measurements.

## Frozen eight-recipe neural roster

Each named component appears once at the stated equal weight; no further subsets, weights, all-nine or raw-selected seed-seven variants are authorized.

1. Original equal3: eta0 + eta01 + eta1, divided by 3.
2. Chan64 equal2: G1 + G3, divided by 2 (legacy joint-selected epochs 75/174).
3. F5: eta0 + eta01 + eta1 + G1 + G3, divided by 5.
4. Primary seed3: archived eta0(seed11 legacy) + seed23 legacy + seed37 legacy, divided by 3.
5. Mixed-selection secondary seed3: archived eta0 legacy + seed23 raw-f-selected + seed37 raw-f-selected, divided by 3.
6. Secondary scratch equal2: scratch native raw-f-selected + scratch MTO raw-f-selected, divided by 2.
7. F7_seed_legacy: (5 F5 + seed23 legacy + seed37 legacy)/7.
8. F7_scratch: (5 F5 + scratch native raw-f-selected + scratch MTO raw-f-selected)/7.

Scratch recipes remain pending their completed-family gates; this review did not inspect them. Descriptor standalone D and fixed half-D/half-F5 are separate approved probes. Original primary contrasts remain unchanged.

## Evidence

The companion SEED_FAMILY_INDEPENDENT_VERIFICATION.json records the independent calculations, exact input hashes and limits; its script is independent_seed_family_review.py. Executor SEED_FAMILY_AUDIT.md and SEED_FAMILY_PROVENANCE.json document the original summary receipt and subsequent prose-only audit corrections without rewriting execution history. Companion provenance binds these artifacts, original RESULTS.json, fixed-seven results, adapter review/receipt and the final eight-recipe amendment. Raw arrays and checkpoints remain server-only.
