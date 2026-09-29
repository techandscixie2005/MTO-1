# Frozen-F round: interpretation and next question

## Outcome

Neither objective produces a material improvement. The trace arm selects epoch2 at native validation R2 .405407830 (+.000113700 from its anchor); raw_f selects epoch1 at .405387525 (+.000093404). The raw_f-minus-trace difference is -.000020306. All are far below the prespecified .003 threshold. Descriptive paired molecule bootstrap intervals include zero; they are conditional on reused validation and checkpoint selection, not fresh confirmation.

At those native-selected checkpoints, fixed historical calibration gives .417538722 and .417865155, both below the calibrated incumbent near .41811924. Keep the existing calibrated eta0 recipe as the strongest eligible deployment reference. No independent-seed promotion is justified for these small effects, and no source predictions were averaged.

The fixed20 native scores are .403125542 and .402157980. Freezing the base greatly limits the deterioration seen in the preceding full-model continuation (.366–.367 at20), but prevention of that larger regression is not an accuracy gain over the unchanged base.

## Optimization and errors

The adapter is active. At epoch20, mean relative change of the right raw M on the fixed TRAIN diagnostic subset is6.046% for trace and4.973% for raw_f; maxima are10.055% and9.194%. No batch triggers gradient clipping. Every original parameter and every buffer remains unchanged. The null result is not explained by a dead adapter, clipping, or accidentally frozen adapter weights.

Online TRAIN normalized raw-f loss decreases from .353728 to .348566 for trace and .353720 to .348143 for raw_f between epochs1 and20. Trace and energy losses also decrease. These are batch means while the model is changing, not end-epoch fixed-model TRAIN evaluations. Validation native-f error ultimately increases despite those TRAIN improvements. The fixed coefficient1 raw_f objective has larger gradient magnitude than trace; no gradient scaling was fitted.

True bright tails improve slightly while false-bright errors grow. At epoch20, q99 true-bright RMSE improves from .219913 to .216994 (trace) and .216822 (raw_f). By contrast, the true-dim/predicted-bright q99-bin SSE grows from the common baseline12.6037 to13.7598 and13.9470. Their complete pooled SSE increases are only .3435 and .4968, so gains in other bins partly offset false-bright regression. Every valid label belongs to exactly one reported bin; no labels were excluded or downweighted for evaluation. The pattern persists under freezing, with much smaller magnitude than the full-model round.

Energy changes through the frozen decoder when its adapted inputs change. Epoch20 energy RMSE improves from .123109eV to .122256 and .122695eV. This confirms that the adapter is not strictly dipole-only even though all original decoder parameters stay fixed.

These results do not distinguish a restricted F function class from a mismatch between in-sample and held-out residual distributions. Both remain possible. Increasing the same budget or learning rate without a new controlled question is not warranted by these results.

## Full-TRAIN calibration-moment supplement

One identity-baseline forward over the existing TRAIN cache evaluated all1,203,550 valid printed-f labels, including22,646 zeros. Only TRAIN label rows were decoded. No optimizer updates, validation/test inference, exported arrays, or deployed calibrator resulted.

- Native TRAIN R2: .6461035600; MSE: .000888627323.
- Unconstrained pooled least-squares affine slope/intercept, computed only as descriptive moments:1.0035907458 and .0000905206.
- Already-fixed historical validation slope/intercept: .8511830211 and .0037251854.
- TRAIN slopes for states1–10: .9779,.9463,.9545,.9445,1.0449,.9752,1.0824,.9792,1.1166,1.1466.

Thus the TRAIN residual distribution favors nearly identity global calibration, and amplification for several higher states, while the historical validation-fitted map shrinks predictions. This provides a concrete reason why an adapter fitted to source-model TRAIN residuals may fail to learn the needed held-out correction. It does not establish causality, and the historical validation coefficients remain exposed estimates. Neither these new TRAIN coefficients nor new validation coefficients were applied or scored.

## One next controlled question

Test whether calibration learned from predictions that are held out from the source model transfers better than calibration learned from its in-sample predictions. Root proposed one fixed internal80/20 split of the existing TRAIN set, without changing outer train/validation/test membership. A label-clean auxiliary original-MTO source would train from random initialization on80% for a fixed33epochs, with normalization from that80% and no calibration-label, outer-validation, or test checkpoint selection.

Both calibration arms should use exactly the same20% labels. One receives auxiliary-source predictions for those held-out molecules; the other receives the current full-baseline predictions for those same molecules. Both ultimately deploy on the same full baseline plus one readout in one complete checkpoint. The auxiliary source is a training aid and is not needed at inference. No prediction averaging is involved.

Start with matched two-parameter global affine moment fits as a cheap diagnostic once the auxiliary source exists. Keep the exact raw anchor and the already-fixed historical affine map as separate benchmarks, marking the latter's validation exposure. Use the same predeclared nonnegativity rule for both new maps. If transfer evidence warrants a richer readout, use small observable E/f inputs with identical capacity, objective, initialization and update budget in the two source arms. Learned h/M features from separately initialized models have arbitrary channel bases and should not be transferred as if aligned.

The main caveat is transfer between an80%-trained source and the full baseline: their prediction errors can differ because of sample size and training, beyond simple in-sample status. A result would test this concrete procedure, not prove a general theory of calibration. The dominant cost is one original-model33epoch fit, estimated roughly1–1.5hours from measured throughput; calibration and cached evaluations are much smaller. Exact protocol, source audit, internal split, matched controls, publication and healthy-GPU admission require a separate decision. No next fit is authorized by this note.

## Verification and records

All64 frozen source/dependency hashes remain unchanged. Both arms complete20epochs with identical order hashes, also matching the preceding round. Receipt/checkpoint/selection/history/array checks pass. Selected and fixed20 geometry replays pass with R2 deviations below1e-7; the selected loader reads one complete checkpoint and no source dataset, cache or original base checkpoint during prediction. Evaluation labels are loaded separately by the evaluator.

All resumable last checkpoints remain on the server with37,620 AMSGrad updates, full selector history and Python/NumPy/Torch/CUDA/order RNG state. CHECKPOINT_INVENTORY.json exports only metadata and hashes. The initial terminal audit import-order error occurred before inference, was corrected, and its log was preserved. No fitted code, settings, source weights or production results were changed.

Primary records: ROUND02_RESULTS.json/.md, ANALYSIS_RECEIPT.json, TERMINAL_REPLAY.json, CHECKPOINT_INVENTORY.json, TRAIN_CALIBRATION_AUDIT.json, ROUND02_CURVES.svg and their sources/logs. Independent review and root's decision precede the required archive-first completion publication.
