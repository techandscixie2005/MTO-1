# Frozen affine eta0 baseline: one model and one checkpoint

This export packages the original eta0 network and the already frozen historical calibration. It performs no fitting, model averaging, checkpoint averaging, or new calibration search. All base weights and the two FP64 calibration buffers are stored in one server-only checkpoint.

## Predictor

`f = max(0, 0.8511830211044088*f_native + 0.003725185373211049)`, where `f_native=(2/3)*(E_eV/27.211386245988)*trace(A)`.

The primary output `f` is calibrated. Returned `E` and `A` are unchanged auxiliary outputs, so calibrated `f` does **not** satisfy the original energy/trace relation with those outputs. No tensor rescaling was substituted. A physically rescaled tensor would be a separate recipe requiring its own specification.

Checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`.

SHA256: `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`. Size6,415,242bytes;1,552,092 model parameters plus2 fixed scalar buffers. No optimizer states. Inference needs geometry/atomic numbers and the repository model code, with optional geometry-derived edges; it does not read another checkpoint or QC labels.

```python
from calibrated_eta0 import load_predictor
model = load_predictor('/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt', 'cuda')
outputs = model(z=z, pos=pos, batch=batch, n=n, edge_index=edge_index)
f = outputs['f']
```

## Validation parity

`CALIBRATED_EXPORT_VERIFICATION.json` records hashes and a complete6,686-molecule/66,860-label validation replay from the reloaded single checkpoint. Calibration applied to historical saved native-f is bitwise identical to the fixed NumPy formula. The real wrapper formula differs by at most1.39e-17; replay versus historical calibrated-f differs by at most4.17e-7 because FP32 base inference can vary slightly by device/scatter order. Replayed refit-validation R²=.4181192453 versus recorded.4181192388; native replay.4052941266 versus.4052941183.

These refit-validation scores use the same validation set that fitted the two constants. Historical fivefold OOF validation R² remains.41665766, and the neural checkpoint itself had already been validation-selected. Historical reused-test R²=.4596672331 was already observed; no test replay was run for this export. The prior study disclosed early test calibration exposure, a ΔR² interval spanning zero, worsened MAE and worsened brightest tails. Exporting one checkpoint does not strengthen those scientific claims.

Reproduce export/parity in a clean output directory using the same code and frozen inputs:

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python export_and_verify.py --physical-gpu 6
```

The script refuses to overwrite an existing export and takes `/tmp/mto_pouter_gpu_6.lock`, verifies GPU health, rejects competing compute and non-idle memory. Initial guard rejections are retained in two logs; GPU6 had a4MiB Xorg graphics process, which was preserved. The corrected guard distinguishes graphics from compute. No process was stopped or reset. Full successful replay took about9seconds, then released the lock.

Only source, this README, verification JSON and logs belong in lightweight archives. Never download or commit the `.pt` checkpoint or prediction arrays.
