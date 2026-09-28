# Deferred retained-residual final20 validation replay

This helper is prepared for one **future** validation diagnostic. It has not replayed the checkpoint. The architecture screen must finish and its round summary must exist before replay. The trained arm's native-f selection remains epoch 0; final20 is a diagnostic checkpoint and must not be promoted based on this analysis.

## Purpose and fixed definitions

The primary prediction is `abs(C_F × E_pred × trace(base_A) + delta_f)`. The trace and energy are promoted to float64 **before** multiplication, exactly as in the architecture evaluator. The float32 forward output is saved only as a numerical sensitivity column. The fixed threshold for a bright target is raw validation `f_true >= 0.0546` (pooled q90). The cross-table partitions physical states 7/8 versus all others, below q90 versus q90-or-brighter, and negative versus nonnegative pre-absolute signed prediction. It compares frozen eta0, the same final20 checkpoint's changed base component, and the emitted final20 prediction. The base comparison is an inference decomposition, not a retraining counterfactual.

For each molecule, squared error sums over all ten states. The report gives top 1, 5, 10, and top 1% concentration, both ranked by final error and by worsening from eta0. All reported raw-f R² uses the pooled validation SST. Raw true f determines diagnostic bins only; it never enters the model.

## Preparation validation already safe to run

```bash
cd /home/inspur/MTO-1
experiments/qm9s_full_EA_20260925/env/bin/python research/oscillator_r2_20260928/completion_receipts/final20_replay.py self-test
```

This verifies a synthetic 8-cell decomposition, absolute-value folding identity, concentration counts, and two validation-only input rows on CPU. It performs no model forward pass and opens no test labels or loader. The result is `FINAL20_REPLAY_SELF_TEST.json`.

## Deferred one-time execution

After all four architecture arms have `FIT_COMPLETE.json`, the architecture round summary exists, and a suitable GPU is healthy and idle, the project coordinator may authorize replay by writing a separate JSON file with:

```json
{
  "approved": true,
  "scope": "retained_residual_final20_validation_replay",
  "gpu": 1,
  "script_sha256": "<sha256 of final20_replay.py>",
  "protocol_sha256": "<sha256 of FINAL20_REPLAY_PROTOCOL.json>",
  "final_checkpoint_sha256": "c96f850f10e739632c90f8cc870990de92fa79dbeed9cfc3e1958d349122ed41"
}
```

`gpu` may be 1, 4, 5, or 6 only if idle, eligible, and under the shared `/tmp/mto_pouter_gpu_<gpu>.lock`. GPU 5 additionally requires the pinned ECC record and numerical microcheck. The helper checks authorization, frozen file and source hashes, the completed checkpoint state, healthy GPU ownership, validation IDs/indices/targets, and recomputed final20 metrics before writing any result. It refuses to repeat if a result already exists.

```bash
cd /home/inspur/MTO-1
experiments/qm9s_full_EA_20260925/env/bin/python research/oscillator_r2_20260928/completion_receipts/final20_replay.py replay --authorization /absolute/path/to/coordinator_authorization.json
```

The small `FINAL20_REPLAY_RESULTS.json` is archivable after review. The `runtime/final20_val_replay.npz` array is retained on the server only and excluded from local D: and GitHub archives. No training, queue, or selected checkpoint changes occur. The script never constructs a test loader or reads test targets or predictions.

