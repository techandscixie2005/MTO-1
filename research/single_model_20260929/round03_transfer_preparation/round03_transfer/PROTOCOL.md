# Round03: internal training holdout for affine transfer

## Question and limits

Does a scalar map fitted to predictions on molecules withheld from a clean source model transfer better to the full baseline than the same map fitted to its in-sample predictions? The two maps use identical calibration labels and the same final full baseline. This is a transfer diagnostic. The historical validation-fitted affine comparator has already used outer validation and is near the best map in this family; merely approaching it is not a new-best or independent-confirmation claim.

The auxiliary source sees fewer molecules and has its own optimization error. A single internal holdout is not full cross-fitting. Any difference combines those distribution differences. The final predictor consists of one original MTO and one fixed coefficient pair in a complete checkpoint, with geometry input only. No auxiliary source, averaging, quantum-chemical labels, or cache is required for deployment. Affine f can be inconsistent with the unchanged predicted E/A tensor; that limitation is explicit.

## Frozen split and clean source

The original outer TRAIN/validation/test partitions remain unchanged. Metadata-only seed20260930 shuffles sorted resolved original TRAIN group keys. Whole groups are reserved until 24,071 rows are reached. Unresolved groups remain in source fitting. Original TRAIN order is retained inside both subsets. All120,355 original TRAIN molecules are assigned once:96,284 source-fitting and24,071 calibration molecules. No label participates in the split and no group is broken. Index arrays remain server-only, with hashes in split_manifest.json.

fit_normalization.json uses only the96,284 fitting molecules: FP64 raw E per-state means, mean centered E variance, mean A squared Frobenius norm, and median atom count. Original FP32 curated E/A are used for optimization, matching the baseline. All962,840 E labels and A labels are valid. No f, calibration, outer-validation or test target is decoded during source initialization or fitting.

Build the original eta0 MTO directly with torch seed11 and these fit-only statistics. No initial_model.pt, full-training learned parameters, or full-training E_state_mean is read. source_config.json is the original architecture config byte-for-byte; config.json supplies this run's fixed schedule. Its source_checkpoint field identifies the full baseline used only in the later affine/export stage.

Train exactly33 epochs on LE+Ls, original normalization and masks, fresh Adam AMSGrad with lr0.001, betas(0.9,0.999),eps1e-8,weight_decay0,batch64,gradient clipping5,FP32,noAMP/noTF32,2CPU threads. Use NumPy default_rng11 permutations of the source-fit subset. No validation, early stopping or checkpoint search occurs. Original ReduceLROnPlateau patience50 cannot reduce within33 epochs, so omitting it preserves the learning-rate trajectory while removing label access. Source selection is epoch33 regardless of source-fit losses.

Atomic last.pt records the entire model, optimizer, RNG, statistics, source manifest and completed history every epoch. An interrupted partial epoch restarts from the preceding complete epoch, with the saved order RNG. A normal finish emits source_final.pt at33 and FIT_COMPLETE.json. No best checkpoint is selected. Resumable optimizer states remain server-only.

## Fixed affine stage

After source completion, predict the same24,071 calibration molecules with fixed epoch33 source and the historical full baseline. Pool every valid printed raw-f label, including zeros. Fit exactly one unconstrained FP64 least-squares slope/intercept for each prediction source. A zero/nonfinite denominator or nonfinite input/coefficient is a reported failure; no fallback, clipping of fit inputs, row exclusions or coefficient search occurs. The post-fit rule for both is f'=max(0,a*f+b).

Write COEFFICIENTS_FROZEN.json containing both pairs and hashes before any outer-validation target access. Resuming an interrupted export reuses that immutable receipt and its hashed predictions, with no refit. Export each pair together with the same entire original full-baseline model. Apply coefficients directly to its NATIVE f, never to the already-calibrated f. The historical affine(alpha0.8511830211044088,beta0.003725185373211049) is a separate exposed comparator.

A separate evaluate command performs one fixed outer-validation comparison after coefficient freezing. One baseline forward pass supplies all four predictors: native, historical affine, in-sample-trained affine and heldout-source-trained affine. This is deterministic transformation of a single prediction, not prediction averaging. Report all66,860 valid raw-f labels, per-state metrics and fixed original TRAIN q90/q99 tail RMSE/MAE/SSE. E metrics describe the unchanged full baseline. Verify both complete exports on the first64 validation geometries; no tuning follows replay. Interrupted attempts preserve a receipt and may repeat only the same frozen computation.

The allocation gate for further nonlinear-readout PREPARATION is heldout-map pooled validation R² at least0.003 above BOTH the in-sample map and native anchor. Per-state and bright-tail tradeoffs remain part of the decision. There is no seed promotion merely for approaching historical calibrated performance, no new historical-test score, and no source-checkpoint search.

## Gates and commands

The bounded preparation smoke checks guarded clean initialization, fit-only access, exact original objective/gradient parity, serialized next-update/order replay, a geometry-only complete-checkpoint loader and known affine fixtures/failure paths. It discards all updates. Independent history review must cover the exact frozen source/config/data hashes. Monitor must first finish round02 publication, then download lightweight round03 records to D:\MTO\archives\ before stage inspection/commit/push. The publication receipt is bound to the frozen manifest and review.

From this server directory, using the pinned environment Python:

```
python prepare_split_statistics.py
python preflight.py --gpu 1
python inference_preflight.py
python freeze_round.py
python launch_round.py --review INDEPENDENT_PRELAUNCH_REVIEW.json --publication-receipt ../ops/ROUND03_PUBLICATION_RECEIPT.json
# After normal source completion, each command independently admits/locks GPU1:
python affine_stage.py fit --gpu 1
python affine_stage.py evaluate --gpu 1
```

The launcher admits a healthy idle GPU with the existing UUID/ECC/remap rules, holds its shared lock, and registers the owned PID/start identity for persistent four-hour monitoring. GPU0 and all unrelated jobs remain untouched. Dataset/index/prediction arrays and all model/optimizer checkpoints stay server-only. Archive code, exact settings, aggregate metrics, logs, hashes, review, analysis and next decision after completion.
