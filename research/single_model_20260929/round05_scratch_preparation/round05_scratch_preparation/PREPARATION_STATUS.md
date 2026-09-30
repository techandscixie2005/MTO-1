# Round05 preparation — independently reviewed and accepted; publication pending

Updated 2026-09-30. Science is the sole model executor; history is the independent reviewer. All preparation stages are terminal. Exactly12 disposable optimizer updates ran on128 new TRAIN molecules. No production fit, validation target access or TEST target access has occurred. Do not repeat statistics, CPU checks, failed GPU attempt, diagnostic, successful retry or Round04.

## Frozen scientific state

- FROZEN_MANIFEST.json:66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1 (82 files). Leave every bound file unchanged.
- Split:c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155; independent verification395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2. Counts120355/6686/6686. Dataset namespace is sealed.
- TRAIN statistics:d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae; actual decoder audit f93e32e63aaa12dc8f81f079e2b96f0d03a5fc5a9235ee98c01ff6fe9e435e02. No complete numeric target array was loaded before slicing.
- GPU preflight:c0a62e24963ed3158f48524fff3f77fae5bce98b91144962b16bb839bcca99a9; independent result review4071eca0211395858f472473abc3891298dc822878e1a76adf248026f5e2c1f5.
- Root amendment234b0ea52edec20bccc703d7ef96429c3ab2705f9da214b7cbde46e2c0e4825a; retry source review d3a1cc9ec7fc771d2546cd53e61ae04b6f8bc61c44b10fd24e67b279c0ceab8e.
- I/O continuity review876e2ea4aec2e401f52b616b25d4881f81f41a636d437ab82880b7f4aa38a70f distinguishes executed training.py17d2bb26... from finalc0147d25...; only two parent-directory fsync append blocks changed. Exact snapshot/mapping/AST proof are frozen. No further model updates or I/O runtime tests were performed.

The first CPU dtype-fixture and GPU bitwise-forward failures remain preserved with their source snapshots. The separate zero-update diagnostic found repeated identical CUDA forwards vary while F/CG on shared raw M is exact. Root explicitly approved the numerical full-forward/export criterion; this was an engineering amendment after diagnosis. Original update/loss criteria and scientific settings stayed fixed.

## Terminal resources and result

Successful retry wrapper1321182/child1321189 exited0 and released GPU1 lock. Registration and exact child identity are in ops/gpu_preflight_attempt_retry01. The final receipt records12 updates, exact RNG/order, model replay max1.1921e-7, optimizer max2.9802e-8, loss difference0, live F gradients and weak fixed penalty. Peak0.62–0.99GB and29.74s wrapper duration are small-fixture engineering measurements, not representative full-TRAIN throughput. All technical tensors remain under private_preflight_retry01 and must never seed production or enter an archive.

## Next actions and blocked production

1. Final independent preparation review passed: INDEPENDENT_PREPARATION_REVIEW.json SHA ef7d90250fd0491fba1432201416de983afa86ee6ae7ebf6e086735f5e10ff19, bound to the exact frozen manifest and complete source_hashes. Final PREPARATION_REPORT.md SHA 1fedd7b8886d00cb5371af62083ddf39e203bde7b8e5a55476fb50a88b0581a1 corrects the authorization-check count wording only; the supplemental review manifest binds that final report. No further numerical tests are planned.
2. Root accepted preparation in research_state/ROUND05_PREPARATION_ACCEPTANCE.md. QC owns the pending publication: download lightweight records FIRST to D:\MTO\archives, verify explicit files and external dependency restoration mapping, independently inspect staged contents, commit and push. Private arrays, IDs, checkpoints, optimizers, caches and all private_preflight directories are excluded. Science remains sole production executor and does not duplicate publication.
3. Only after that verified publication may root issue a distinct PRODUCTION_EXECUTION_AUTHORIZATION.json. PRODUCTION_AUTHORIZATION_TEMPLATE.json is deliberately authorized:false and is not a launch permit.
4. The future command is `/usr/bin/python3 launch.py --authorization /absolute/path/PRODUCTION_EXECUTION_AUTHORIZATION.json` from this directory. It admits control/F/decor/both sequentially onto physical GPUs1/2/4/6 and preserves a separate lock for each concurrent worker. Each child is UUID-prebound before imports and registered by PID/start/boot/cwd/argv. A failed later admission does not stop already admitted owned workers. No unrelated job is signaled.

The fixed four-arm recipe is60 epochs, seed/order11, original LE+Ls, Adam AMSGrad LR.001, batch64, WD0, clip5, FP32/no AMP/TF32, penalty.001 for decor/both. Exact inherited initialization and all60 orders are frozen. Each arm has112860 updates. Epoch0 is eligible; earliest minimum pooled validation raw-f SSE selects one checkpoint. At60, at least.003 R² over contemporaneous control is required to consider a later common continuation to100. No TEST scoring is authorized.

last.pt is the completed-epoch automatic recovery point, with optimizer/RNG/order/history and immutable selected-version hashes. best.pt retains selected optimizer/RNG snapshot; geometry_best.pt will be the self-contained one-checkpoint predictor. Technical state cannot enter this path. New TEST contains5989 oldTRAIN+361 oldVAL+336 oldTEST rows: new disjoint internal partition, not external fresh evidence. No accuracy gain is claimed from these preparation checks. PREPARATION_REPORT.md and PROTOCOL.md contain complete rationale, limitations and commands.
