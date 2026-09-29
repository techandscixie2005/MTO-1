# Prespecified fixed-calibration secondary comparison

Frozen by root before the four-arm pilot starts. The main checkpoint selector remains minimum pooled native raw-f validation SSE, with epoch0 eligible and earliest exact ties retained.

For each arm's native-selected checkpoint, also report `max(0, α*f_native+β)` using **the same historical constants** α=.8511830211044088 and β=.003725185373211049. Do not fit constants, choose among calibration variants, select a different checkpoint using calibrated scores, or inspect test results. Report pooled raw-f R²/SSE/MAE/RMSE and the same state/train-cutoff bright slices, with all valid labels retained. Applying this already frozen map does not average predictions and can be exported as a single model/checkpoint.

The appropriate native anchor is eta0 validation R²=.405294118341. The strongest eligible fixed-calibration baseline has recorded **refit-validation** R²=.418119238799; its historical **out-of-fold** validation R²=.41665766 is a different quantity. Compare the proposed fixed-map result with the fixed-map baseline on the same full validation labels. Do not claim a new architecture exceeds the strongest eligible single-network recipe solely because native R² exceeds.405294.

The historical map was fitted on this validation set, whose neural predictions were themselves checkpoint-selected there. Both native and calibrated development comparisons remain conditional on reused validation. The existing reused-test calibration result is context only. No fit or test evaluation is part of this secondary policy.

For deployed fixed-map outputs, document that f is the calibrated primary output and original E/A are auxiliary; the original f–E–trace(A) identity is not retained. Do not silently rescale A or redefine the model.
