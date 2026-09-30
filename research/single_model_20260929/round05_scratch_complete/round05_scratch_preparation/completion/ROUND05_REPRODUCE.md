# Round05 reproduction and recovery records

All four runs are complete. The commands below document the executed recipe; completed-run guards intentionally reject a rerun in the existing namespace. A new fit, extension or seed requires a new root decision and reviewed authorization. No TEST evaluation is part of this recipe.

## Source and environment

- Repository preparation commit: `9de80e37e1586c2d3068b188ed2c242ee9883147` on main and the research branch.
- Server directory: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation`.
- Frozen82-file manifest: `66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1`.
- Independent preparation review: `ef7d90250fd0491fba1432201416de983afa86ee6ae7ebf6e086735f5e10ff19`.
- v2 split: `c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`; independent reconstruction: `395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`.
- TRAIN statistics: `d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae`.
- Actual root execution authorization: `07308b113557d2196cefc951a603f3f1a5093009eaae1576412f1e393bb8162c`.
- Verified preparation publication receipt: `fbdcbb4ce7e2b86ae6c906affeb4e7c08a57fc9e48e58a60d485dd61c7a4eba5`.

Use Python3.10.19 at `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. ENVIRONMENT.json pins torch2.5.1+cu121, e3nn0.4.4, NumPy2.2.6, scipy1.15.3, sympy1.13.1, torch-cluster1.6.3+pt25cu121, torch-scatter2.1.2+pt25cu121 and torch-geometric2.7.0. The preparation archive contains exact lightweight dependency snapshots/restoration mapping. Restore the pinned paths/code rather than silently substituting later versions. Raw datasets/private split arrays and checkpoints must remain available separately on the server; they are intentionally absent from Git.

## Executed fit command

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /usr/bin/python3 launch.py \
  --authorization /home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json
```

The launcher verifies all source/authorization/review/publication/split bindings, then sequentially admits the four concurrent workers under separate shared GPU locks. Each child inherits its GPU UUID before Python imports. Physical assignment was control1, adapter2, decorrelation4, both6:

| Arm | CUDA_VISIBLE_DEVICES |
|---|---|
| control | GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 |
| adapter | GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa |
| decorrelation | GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184 |
| both | GPU-2431a641-8045-a10b-4aa0-2769010e0708 |

Logical CUDA device0 therefore maps to the assigned physical GPU. GPU0's unrelated job and unhealthy GPUs3/7 were excluded. Owned launch identities, resource XML, exact commands, environment and logs are in `runs/<arm>/attempts/<registered-attempt>/`. Every arm has one original attempt and no resume or failure marker.

Fixed common settings: original MTO16/query32/router128/head128, fresh seed11 and independent order seed11, new TRAIN normalization, original LE+Ls, Adam AMSGrad LR.001 fixed, batch64, WD0, clip5, noAMP/noTF32,60 epochs. Right-F adds4016 trainable parameters when enabled; all arms retain the same F checkpoint schema and controls freeze those parameters. The raw-M penalty is .001 for decorrelation/both and zero otherwise. Neither objective nor schedule adapts to validation. Epoch0 is eligible; the earliest minimum all-label validation raw-f SSE selects the checkpoint.

## Completed artifacts and one-checkpoint inference

Each `runs/<arm>/FIT_COMPLETE.json` binds112860 updates, history, last.pt, best.pt and geometry_best.pt. `last.pt` contains the committed final model, optimizer and RNG/order state. `best.pt` is the chosen training snapshot. `geometry_best.pt` contains the complete geometry predictor and TRAIN config/stats; it is the deployment artifact. No prediction averaging is used.

The v2 reference is:

```text
/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt
SHA256 e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e
Selected epoch45; native validation pooled raw-f R²0.44716940136585204
```

The existing `predictor.load_predictor(checkpoint_path, device='cpu')` reconstructs that one checkpoint with strict model/buffer fingerprints and accepts atomic numbers, positions, batch membership, graph edges and molecule count. Native f is `(2/(3*27.211386245988))*E_eV*trace(A)`. No QC label, original checkpoint, data cache or external statistics is required at inference. Source dependencies remain ordinary software requirements. This closeout performed no inference; the preserved technical preflight verified load/forward access and symmetry behavior.

## Executed terminal analysis

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python terminal_analysis.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /usr/bin/python3 write_completion_report.py
```

The completed analysis guard refuses a second execution. It reduced only saved selected/final validation outputs, checked CPU checkpoint metadata/tensors, and read validation identity-component metadata for descriptive bootstrap. No raw dataset target member, TEST target, or model forward was opened. The exact analysis settings and2000-draw seed20260930 component bootstrap are in TERMINAL_ANALYSIS_CONFIG.json. All66860 labels remain included. Terminal logs, source hashes and aggregate outputs are in ANALYSIS_RECEIPT.json and the terminal manifest.

Reproducing an analysis in a new output workspace requires the same private saved arrays and hashes; remove neither completion markers nor evidence to rerun the existing instance. Recovery from a genuine future incomplete run is separately authorized and starts from its atomic last.pt, never disposable preflight state. These four completed workers require no recovery.
