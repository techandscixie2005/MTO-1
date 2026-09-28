# Validation-fitted calibration of completed MTO η=0 predictions

## Purpose and status

Hypothesis: a small correction fitted to saved validation oscillator-strength predictions can reduce pooled native-f squared error without retraining. This is an **exploratory diagnostic** on the completed MTO η=0 model. The historical test split had already been analyzed in earlier work. Before this scripted round, a read-only calculation also exposed test outcomes for global scaling, global affine calibration, and unregularized per-state affine calibration. Those preliminary test results were disclosed to the research coordinator before the cross-validation protocol was written. This round cannot be treated as a blind test.

## Baseline and data

Source run: `experiments/qm9s_eta_Ef_20260926/runs/mto_eta0`, best epoch 33. Its checkpoint was selected by validation training loss; the η=0 run was selected among η variants by validation spectrum MSE. The saved native prediction uses `f=(2/3)*(E_pred_eV/27.211386245988)*tr(A_pred)`. The target is the original printed `oscillator_strength`, without clipping or outlier removal. All 6,686 molecules × 10 states are valid. Molecule IDs, labels, predictions, and masks come from the saved validation and test NPZ files.

The exact input hashes are recorded in `validation_crossfit.json` and `exploratory_test.json`. The validation NPZ SHA256 is `25acae64bb4e3e541048a8705045f7f639d1113d537dbbef71672752e8065890`; test NPZ SHA256 is `8ad125c55200c55de7c8873ab3261651d6025c45fd48d1b26751b30819bc515b`.

## Method and selection

`calibrate.py` assigns each molecule to one of five deterministic folds by SHA256(ID) modulo five; all ten states stay together. It fits each candidate on four validation folds and predicts the fifth. The fixed family is identity, nonnegative global multiplicative scale, per-state multiplicative scales blended toward the global scale by 25%, 50%, 75%, or 100%, and a global affine map clipped at zero. We select the lowest pooled out-of-fold native-f SSE, then refit that method on all validation molecules. No neural weights or source predictions are changed.

The cross-fit report and frozen choice were written before the script opened the test NPZ. The validation report SHA256 is `be80a185ffa436ecd4770c3735d92d05cb877360b31e1aef433066fff2eb42a3`; frozen choice SHA256 is `dc1ab25eb87b05499fdb0dbe7d50dbcc6d7cd1a6e2ed16b5eeaf30de9616aee2`. Script SHA256: `7ba2de112aae8deaebe3d157840509d61143b390b6a95cb705946943700ab231`.

| Candidate | Validation out-of-fold pooled f R² |
| --- | ---: |
| Identity | 0.405294 |
| Global scale | 0.412808 |
| Per-state scale, 25% local | 0.412903 |
| Per-state scale, 50% local | 0.411975 |
| Per-state scale, 75% local | 0.410023 |
| Per-state scale, 100% local | 0.407048 |
| Clipped global affine | **0.416658** |

Selected map: `f_cal=max(0, 0.8511830211044088*f_native + 0.003725185373211049)`.

## Exploratory test result

| Metric | Native baseline | Selected calibration |
| --- | ---: | ---: |
| Pooled f R² | 0.455815 | **0.459667** |
| f RMSE | 0.039286 | 0.039147 |
| f MAE | **0.017716** | 0.018134 |

Pooled ΔR² is +0.003852. A paired bootstrap resampling 6,686 molecules with all ten states together (2,000 replicates, seed 20260928) gives a 95% percentile interval for ΔR² of **[-0.012204, +0.022829]**. For truth f≥0.0546, the validation-defined 90th percentile, calibrated test SSE worsens by 5.238; at f≥0.2377, the 99th percentile, SSE worsens by 4.852. Per-state detail and all SSE values are in `exploratory_test.json`. That file's SHA256 is `2b882f4557bbc08f5954f7f7a6d0337c78ae1ebd7fdc16853bdd7a41848fa7f8`.

## Conclusion and next action

The correction raises pooled R² slightly, but its uncertainty includes a loss and it damages bright transitions. It is a diagnostic, not evidence of a reliable improvement. The stronger next experiment is training with an objective aligned to native f while retaining an energy loss, with checkpoint selection on validation native-f SSE/R². Keep this result in the historical record and avoid further calibration tuning on the reused test split.

