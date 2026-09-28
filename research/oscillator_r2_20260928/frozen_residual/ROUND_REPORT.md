# Frozen eta0 residual head — completed validation round

**Question.** Can an additive oscillator-strength correction help when eta0's representation, energy prediction, and original tensor-derived base are held fixed? This was one seed-11, 20-epoch head-only experiment on the frozen train/validation split. The historical eta0 raw native-f validation R² was 0.405294118. The previously completed joint residual arm selected epoch 0 and fell to 0.304938 by epoch 20.

**Setup.** The exact sealed width-32 residual head began with a zero final layer. Only its 4,161 parameters were optimized using raw-f MSE divided by the train f variance; eta0's 1,552,092 parameters and all source buffers were frozen. Training used 120,355 molecules, batches of 64 with all ten states, fresh Adam AMSGrad at 1e-4, 1,881 updates per epoch, and 37,620 total updates. Native validation f was computed from the original FP64 tensor-derived base plus the learned FP32 correction, then absolute-valued. Selection used the lowest pooled raw-f validation SSE with epoch 0 eligible. The frozen train/validation feature cache, code, order, GPU health, and completion markers passed integrity checks. No test data were evaluated.

| Checkpoint | Train raw-f R² / SSE | Validation raw-f R² / SSE | Validation E MAE |
| --- | ---: | ---: | ---: |
| Initial, epoch 0 | 0.646104 / 1069.5074 | 0.405294 / 94.20512 | 0.091388 |
| Selected, epoch 2 | 0.647818 / 1064.3256 | 0.406790 / 93.96818 | 0.091388 |
| Final, epoch 20 | 0.652201 / 1051.0807 | 0.404299 / 94.36274 | 0.091388 |

The selected gain over eta0 is **ΔR² = +0.001496** (0.252% SSE reduction). Paired 95% connectivity-group bootstrap interval: [+0.000829, +0.002185], with 100% positive draws in 2,000 replicates. The fixed equal-three eta0/eta01/eta1 ensemble remains higher at validation R² 0.449421; selected frozen-head ΔR² versus it is −0.042632 (group interval [−0.061424, −0.025336]).

The selected head reduces SSE by 0.5160 on raw true f at or above validation q90=0.0546, while raising SSE by 0.2790 below q90. State 7 loses 0.0133 SSE and state 10 gains 0.0790. Molecule ID 14562, identified before this run in the joint-residual replay, worsens by only 0.0122 SSE at the selected checkpoint; it worsens by 0.5609 at final epoch 20. Full validation SSE improves by 0.23694 at the selected checkpoint, so the single-molecule example is illustrative rather than the decision metric.

**Conclusion.** Freezing the base avoided the large degradation seen in joint residual training, but the selected gain is below the preregistered +0.005 tentative threshold. Train R² kept rising while validation peaked at epoch 2 and ended below eta0. Retain this as a small exploratory diagnostic; do not promote the model, inspect the historical test, extend training, or infer a fundamental backbone limit. Any further research direction requires a separate reviewed hypothesis.

**Records.** `INDEPENDENT_RUN_AUDIT.json` verifies the run; `FROZEN_RESIDUAL_RESULTS.json` contains all 21 full-train and validation checkpoints and bootstrap results; `FROZEN_ERROR_DECOMPOSITION.json`, `FROZEN_VS_ENSEMBLE_VALIDATION.json`, and `WORSTCASE_14562_VALIDATION.json` contain the fixed diagnostic slices. The validation split was reused for checkpoint selection, and all intervals are conditional on that selection and one seed.

