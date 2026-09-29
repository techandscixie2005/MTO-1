# Round03: held-out-source affine transfer

## Outcome

The held-out-source map improves pooled validation raw-f R² by **0.011080390 over the native baseline** and **0.014350164 over the matched in-sample map**. It passes the prespecified gate for considering nonlinear-readout preparation. It remains below the exposed historical affine incumbent by 0.001744729, so this round establishes no new best predictor and no independent-seed promotion. The root decision retains that incumbent; no new fit follows automatically.

| Predictor applied to the same full baseline | Raw-f R² | MAE | True q99 RMSE | False-bright count | False-bright SSE |
|---|---:|---:|---:|---:|---:|
| Native | 0.405294124 | 0.017785608 | 0.219912917 | 112 | 12.60367458 |
| In-sample affine | 0.402024351 | 0.017819879 | 0.218778366 | 123 | 13.32050613 |
| Held-out-source affine | 0.416374515 | 0.017805840 | 0.227281574 | 86 | 9.71685855 |
| Historical validation-fitted affine | 0.418119244 | 0.018219066 | 0.231384276 | 69 | 8.41899201 |

All 66,860 valid outer-validation labels are retained, including zeros. q99 is the fixed original TRAIN cutoff 0.2412; q90 is 0.0549. False bright means true f below q99 and predicted f at least q99. These bins depend on each predictor and therefore change membership; their differences are descriptive rather than a fixed-row causal decomposition. The four bins exhaust every label and their SSE sums match pooled SSE.

The held-out map improves R² for states 2, 3, 4, 6, 7, 8, 9 and 10; state 1 slightly worsens (0.827284→0.827166), and state 5 worsens (0.420834→0.416277). Its true q90 RMSE worsens from 0.095359 to 0.097486, q99 RMSE worsens from 0.219913 to 0.227282, and MAE increases slightly. The pooled improvement does not remove these tradeoffs. Complete per-state counts/R²/SSE/RMSE/MAE and q90/q99 counts/SSE/RMSE/MAE are in ROUND03_RESULTS.json; ROUND03_RESULTS.md provides the aligned state table. Energy is unchanged by either map: validation RMSE is 0.123109392 eV.

## Source and coefficient quality

The clean source used 96,284 fitting molecules, original MTO/LE+Ls, seed 11, fixed 33 epochs and lr 0.001 AMSGrad. All 33 prescribed orders and 49,665 optimizer updates were reproduced in the terminal audit. Its complete-epoch compute time was 3921.72 s (65.36 min); no held-out/outer label selected the source. Online source-fit total loss decreased from 0.368548 to 0.121808, with E loss 0.134355→0.027771 and trace loss 0.234193→0.094036. These are losses of changing models, not final-model TRAIN evaluations. Resumable last.pt retains 135 optimizer states at 49,665 steps and all RNG states.

Both coefficient fits use the same 240,710 valid raw-f labels from 24,071 reserved molecules, including 4546 zeros. Native predictions on those molecules have R² 0.429408124 for the held-out source and 0.661066543 for the full baseline that trained on them. Source energy RMSE there is 0.134598291 eV. The source's smaller training population and different optimization errors are real confounders; this comparison tests one transfer procedure and does not establish the cause of earlier adapter failures.

Apply exactly one map to the full baseline's native f:

```
held-out: f_out = max(0, 0.9025853252242239 * f_native + 0.0018285582112617521)
in-sample: f_out = max(0, 1.0180950011215777 * f_native - 0.0002909346448593461)
```

Both maps were frozen before the sole validation attempt. They are not composed with the historical map. No source predictions or checkpoints are averaged. The auxiliary source is absent from final inference; each exported checkpoint contains the same original full baseline and one pair of coefficients. Affine f need not equal the unchanged auxiliary E*trace(A). See ROUND03_REPRODUCE.md for exact paths, hashes and device-bound commands.

## Evidence limits and resource record

Validation has prior historical exposure and is not fresh holdout confirmation. The historical affine comparator was fitted using that validation information. No test inference, nonlinear head, new architecture or independent-seed confirmation occurred in this round.

The source fit's registered GPU1 process completed and exited; source_final/last hashes match its terminal receipt. The short affine-fit interval included a monitor observation of an MTO Python process allocating 504 MiB on GPU0. Its argv and full inference placement were not captured before exit, so the fit cannot be claimed to have run wholly on admitted GPU1. Outputs were preserved without a repeated fit. The one validation process was launched with GPU1's UUID before Python imports, positively observed only on GPU1, and exited with code 0; both coefficient/export hashes stayed unchanged. No process was signaled or unrelated job modified. RESOURCE_PLACEMENT_NOTE.md retains the uncertainty and exact observation.

Independent terminal integrity checks passed: all 69 frozen hashes; split/stat/initialization/order contracts; final/last model equality; full-base tensor/config/stat identity in both exports; coefficient replay; freeze-before-attempt chronology; all reported metrics/bins; and observed validation placement. Independent scientific review and the root decision are separate closeout records. Lightweight records must be archived to D:\MTO\archives\ before inspected commit/push. All weights, optimizer states and arrays stay on the server.
