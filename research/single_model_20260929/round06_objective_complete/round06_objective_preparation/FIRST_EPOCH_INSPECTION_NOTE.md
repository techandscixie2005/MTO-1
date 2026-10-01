# Round06 first committed epoch inspection

Both actual histories/statuses showed epoch1 complete and no FAILED marker before the read-only inspector ran once. It exited0 and produced `ops/FIRST_EPOCH_METADATA.json`, SHA256 `5e74e06672684e76b68ff33fbbc146628c19e100eeb77ba226daf0aac9d3559f`. The source is `inspect_first_epoch.py` SHA `4a2499f3550c005eb1b9951f1a3577764d9126ec42bbe1aaa436a12c40e75bb9`, accepted by independent continuity review117f1a4c. It changes only the arm tuple and resumable-format tag from the earlier inspector.

The inspector opened one atomic committed last.pt inode per arm and CPU-loaded its tensors to check metadata. Both contained1,881 Adam steps,135 optimizer entries, AMSGrad moments, all five RNG categories, exact initial tensor/order/manifest/authorization bindings and valid immutable selected/prediction hashes. TRAIN and validation numeric-decode ledgers match the exact frozen partitions; TEST numeric rows0. No model was constructed, no model inference or raw-target array decoding occurred, and no production file was modified. The checked checkpoint hashes identify the observed epoch1 snapshots; later last.pt commits naturally replace them.

First-epoch durations were155.429607629776s control and160.6291468143463s raw_f. Fixed60 extrapolation is about2h35–2h41 per concurrent arm, excluding changing load and startup. Early validation scores remain descriptive; no objective, learning rate, selection, budget or allocation rule changes follow. The workers continue naturally; no further live polling or metadata rerun is needed before the scheduled monitor.

Command already executed, not a pending action:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python inspect_first_epoch.py \
 > ops/FIRST_EPOCH_INSPECTION.log 2>&1
```
