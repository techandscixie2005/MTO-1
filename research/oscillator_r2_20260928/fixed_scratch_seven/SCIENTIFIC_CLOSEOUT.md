# Fixed scratch-seven independent result review

Recorded at 2026-09-29T08:13:44.026590+08:00. **PASS** for exact saved-array arithmetic, provenance and all prescribed paired comparisons. Close the final neural mixture without extra weights, subsets, inference or test access.

F7_scratch is exactly the equal native-f mean of eta0, eta01, eta1, G1, G3 and the raw-f-selected native/MTO scratch models. All seven individual weights are 1/7 and eta0 appears once. The selected scratch checkpoints and both legacy-selected seed reference checkpoints match genuine 100-epoch/188100-step terminals and receipt hashes. All validation IDs, raw FP64 labels and authoritative all-valid masks align. No model forward was run by this audit.

| Predictor | SSE | Pooled raw-f R² |
|---|---:|---:|
| Frozen F5 | 81.151578 | 0.487699605 |
| F7_seed_legacy | 81.311355 | 0.486690951 |
| F7_scratch | 82.277078 | 0.480594453 |

Versus F5, F7_scratch loses 1.125500 SSE (ΔR² −0.007105152). The independently reproduced paired group interval is [−0.012543478, −0.002559570], positive fraction 0.002; molecule interval [−0.012759, −0.002402]. Eight states worsen; only S9/S10 improve. Below-q90 SSE worsens 0.250158, q90 bright 0.875342 and q99 0.263819. Across molecules, 3,095 improve and 3,591 worsen: gross gains 2.597241 versus losses 3.722741. Largest gain is ID3026 (+0.099748 SSE); largest loss is ID14562 (−0.259132, 2.622175→2.881307 SSE). The six lowest-q2 cases worsen collectively from 4.141298 to 4.626043. Absolute totals, rather than positive gain shares, are appropriate for this negative net result.

Versus F7_seed, ΔR² is −0.006096498, group interval [−0.017292030, +0.010563390], positive fraction 0.1825. F7_scratch improves below-q90 SSE by 0.883754 but worsens q90 bright by 1.849477 and q99 by 0.772682. Case14562 improves by 1.060892 and six low-q2 cases by 0.870139; these local gains do not outweigh losses elsewhere. Tiny geometry strata do not establish geometry causality or justify case filtering.

Both bootstraps use exactly 2,000 paired draws, seed20260929, sorted connectivity groups and all ten states with pooled SST recomputed. All state/tail/signed-error/q2/concentration totals reproduce independently. Seven forwards versus F5's five add inference cost without improving this fixed comparison; counts are not measured latency. This does not prove weak single models or scratch diversity can never help. The mixture was fixed post-native-result and before completed paired outcome review, so its uncertainty is descriptive under adaptive reused-validation selection, not fresh confirmation or architecture causality.

Root selected F5 under the previously recorded maximum pooled validation R² policy after closing all approved candidates. Any later historical-test comparison requires a separate immutable candidate freeze, reviewed protocol/code and explicit authorization; this note authorizes none. INDEPENDENT_RESULT_REVIEW.json and independent_result_review.py preserve independent checks; SCIENTIFIC_PROVENANCE.json binds exact source/result/review bytes. Runtime prediction arrays and checkpoints remain server-only.
