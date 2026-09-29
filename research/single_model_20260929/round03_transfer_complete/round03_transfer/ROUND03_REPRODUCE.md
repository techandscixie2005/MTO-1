# Round03 recipes and reproduction

All paths below are on USTC-A800. The pinned executable is `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. Work directory is `/home/inspur/MTO-1/research/single_model_20260929/round03_transfer`. Package/source versions and exact source hashes are recorded in the parent ENVIRONMENT.json and this round's FROZEN_MANIFEST.json (SHA93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37).

## Complete single-checkpoint predictors

| Recipe | Checkpoint | SHA256 |
|---|---|---|
| Held-out-source affine | `affine/heldout_source.pt` | `291b80c87afc0c841781d3c7e1b376f6008a13d955291110d5305280d75df398` |
| Matched in-sample affine | `affine/in_sample.pt` | `3f618f960492af279c6d569ee5b79cf1b43af86ba48d5d4fd53b38cb66d63387` |
| Fixed epoch33 auxiliary source, training aid only | `runs/source33/source_final.pt` | `769026edf4e999799dd201c3e8f2ad3393e25ef9b3a5a99b23f7ce59ffc984ec` |
| Source resumable state | `runs/source33/last.pt` | `52f2884b4869ae87d85ba56875021ff3d42383cd53bfad356362d213883499c9` |

Prefix each relative checkpoint path with the work directory above. The two deployment checkpoints each contain exactly the original full baseline's tensors, original configuration/statistics, nonpersistent-buffer provenance, and their own alpha/beta buffers. `predictor.load_predictor(path, device='cpu')` loads one complete file. It accepts geometry-derived z/pos/batch/n/edge_index keyword tensors in the original MTO graph convention and returns E, A, native_f and calibrated f. No labels, original source checkpoint, training arrays, cache or auxiliary source is read by the loaded predictor. The original graph preparation and tested geometry fixture are in clean_source.py/inference_preflight.py; positions follow the original Å convention.

Example loader usage (with geometry prepared separately):

```python
import torch
from predictor import load_predictor
model = load_predictor('/home/inspur/MTO-1/research/single_model_20260929/round03_transfer/affine/heldout_source.pt', device='cpu')
with torch.inference_mode():
    outputs = model(**geometry)
    oscillator_strength = outputs['f']
```

The strongest eligible incumbent remains `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`, SHA `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`, using its own baseline loader. Its historical validation R²≈0.41811924 remains above the Round03 held-out map≈0.41637451. These have different calibration/validation exposure and neither is fresh holdout confirmation.

## Fixed source and affine commands

The exact scientific settings are in config.json, source_config.json and PROTOCOL.md. prepare_split_statistics.py deterministically recreates the 96,284/24,071 internal split while preserving groups, and source-only normalization from the pinned original TRAIN files. Source fitting uses fresh seed 11 weights, original LE+Ls, fixed 33 epochs, AMSGrad lr 0.001, batch 64, clip 5, weight decay 0 and FP32. No source selection on reserved/outer labels occurs.

The completed directories are immutable evidence. These commands describe reproduction or authorized recovery; do not rerun completed fitting/scoring to choose a preferred result. A new replication needs its own namespace/review/resource/archive gates.

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round03_transfer
MTO_PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
export CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES= "$MTO_PY" prepare_split_statistics.py
"$MTO_PY" preflight.py --gpu 1
CUDA_VISIBLE_DEVICES= "$MTO_PY" inference_preflight.py
CUDA_VISIBLE_DEVICES= "$MTO_PY" freeze_round.py
# Independent review and archive-first publication must pass before launch.
"$MTO_PY" launch_round.py --review INDEPENDENT_PRELAUNCH_REVIEW.json --publication-receipt ../ops/ROUND03_PUBLICATION_RECEIPT.json
```

The source launcher prebinds physical GPU1's UUID in the child environment, performs health/shared-lock admission and registers the owned PID. After fixed 33-epoch completion and process exit, verify_source_gate.py checks all frozen hashes and final/last checkpoint receipts. The affine commands must also prebind the UUID before Python starts:

```
CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 PYTHONUNBUFFERED=1 \
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py fit --gpu 1

CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 PYTHONUNBUFFERED=1 \
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py evaluate --gpu 1
```

Physical GPU1 is the sole visible logical CUDA device 0; `--gpu 1` remains the physical admission/lock index. The actual one-time validation used reviewed stdlib `run_validation_bound.py`, which creates that child command/environment and records its PID/start identity, actual UUID observations and exit status. Its receipt is VALIDATION_LAUNCH_20260929T200405597244Z.json. The fit's original invocation lacked external prebinding; RESOURCE_PLACEMENT_NOTE.md documents the observed GPU0 process and uncertain full fit placement. Do not silently rewrite that execution history or repeat its frozen coefficients.

`fit` resumes a partial export from the existing immutable coefficient receipt and hashed prediction arrays, without refitting. `evaluate` refuses when VALIDATION_COMPLETE.json exists. CPU-only `summarize_transfer.py` validates existing records and regenerates aggregate reports; it performs no model inference. Store raw/index/prediction arrays and all checkpoints only on the server. Archive only lightweight code/settings/logs/receipts/aggregates/reviews and decisions, first to D:\MTO\archives\, then through inspected commit/push.
