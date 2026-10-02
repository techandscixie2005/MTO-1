# Round08 preparation accepted — publication pending

Sole executor: /root/science_implementation. Root accepted the engineering preparation in ROUND08_PREPARATION_ACCEPTANCE.md, SHA256 17efd5b82d201401c5d18cdf40bc76bb318be7799289481beaf16e501eb3bf61. QC owns the D-first archive and publication; history independently reviews the inventory and staged bytes. Production remains blocked until verified publication and separate bound root execution authority. No completed numerical stage may repeat.

## Sealed evidence and limits

- FROZEN_MANIFEST.json: de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146, exactly 258 files.
- INDEPENDENT_PREPARATION_REVIEW.json: 87de2d873f2a24033a2f646ac825f26abbbae7eadbd5305a723fbb6a17093e8b.
- Nine-member INDEPENDENT_REVIEW_MANIFEST.json: 4ee5e4d010e0e4c689562adcea2b0e6b1a4d07dff70b4067e0bb8178f039f99d.
- PREPARATION_REPORT.md: edf708c8530f51d1447cdbb68971c19d42c157e4f749f91a804fb34da6999eb1.
- Frozen RUNNER_HANDOFF.md: f2a9bed9121b2ccc7a6a834a4aa2ba8eb950fe1424e48160ca7cdde64158711c.

CPU checks ran once and passed. The Python child exited 0; the separate outer-shell CRLF exit-token error is preserved and must not trigger a repeat. GPU_PREFLIGHT.json 9254d560275f8c6a2d7c2e47ea249ae999a7e4855d84be81be4821174d191acf records exactly nine discarded updates on 128 TRAIN molecules, with zero validation/TEST numeric rows. GPU review e9be9f22831b5bf55f09c87b05dbceb8fb8af1455964656c3f6afb54311090a4 passed. The registered child 1649927/start1280710605 exited normally; its GPU1 lock was released. All technical states remain server-only and must never initialize production.

Transport gradients and early relative effects were very small. Technical liveness does not establish useful learning or accuracy. Root explicitly preserves the fixed residual scale, coefficients, optimizer and LR; no tuning follows these diagnostics. Preserve the registration-before-compute correction and the reviewer's transient R/S process-state comparison correction, without changing sealed records. The goal .60 remains unmet; retained v2 reference remains Round05 control45 .44716940136585204.

## Future authority schema — not issued by this handoff

Server namespace: /home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation.

Root must separately write PRODUCTION_EXECUTION_AUTHORIZATION.json with these exact fields:

```json
{
  "authorized": true,
  "scope": "round08_three_arm_60epoch_fit",
  "frozen_manifest_sha256": "de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146",
  "independent_review_sha256": "87de2d873f2a24033a2f646ac825f26abbbae7eadbd5305a723fbb6a17093e8b",
  "publication_receipt": "/home/inspur/MTO-1/research/single_model_20260929/ops/ROUND08_PREPARATION_PUBLICATION_RECEIPT.json",
  "publication_receipt_sha256": "ROOT_MUST_BIND_THE_VERIFIED_RECEIPT",
  "split_manifest_sha256": "c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155",
  "independent_split_verification_sha256": "395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2",
  "epochs": 60,
  "arms": ["original", "local", "neighbor"],
  "test_access": false,
  "historical_weights": false
}
```

This is documentation, not a saved authorization. Preparation acceptance or publication does not release the execution gate. The launcher verifies exact source/review dictionaries, every source hash, split/verifier pins and the archive-first/remote-verified publication receipt before mutation.

## Future command — currently blocked

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  launch.py --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms original local neighbor
```

Admission is sequential, then the three registered workers run concurrently: original on physical GPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800; local on GPU2 UUID GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa; neighbor on GPU4 UUID GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184. Each needs fresh healthy/idle admission and its exclusive /tmp/mto_pouter_gpu_<index>.lock. UUID visibility is bound before scientific imports. The registered_entry.py pipe barrier releases the same owned PID only after registrar success bound to the authority hash. Registration failure stops only the owned child and preserves failure; there is no automatic retry or GPU fallback. Do not touch unrelated GPU0 jobs.

## Recovery and terminal rules

The per-arm last.pt is the committed completed-epoch resume point, with full optimizer/AMSGrad, RNG/order, history and source/split bindings. best.pt is a selected snapshot, not this runner's committed resume source. An interrupted epoch replays from the preceding committed boundary; numerical parity is bounded, not a guarantee of bitwise CUDA trajectories. All private checkpoints, raw/prediction/identity/split arrays and caches remain on the server.

An app interruption is not a worker failure. Inspect exact registered identity, status, failure and terminal markers before acting. Preserve any real failure; recovery needs an explicit reviewed decision with the unchanged source/settings, a fresh admission and a distinct attempt/registry ID. Reuse the same launcher only for the specifically authorized affected arm. Completed FIT_COMPLETE arms refuse rerun. Report partial launches instead of cancelling healthy admitted workers or silently starting replacements.

Completion requires 60 epochs, 112860 updates, all 60 prescribed orders, unchanged source/split, opaque checkpoint/export hashes and zero TEST access. Selected checkpoint is the earliest native-validation SSE minimum over epochs 0–60. Later saved-output analysis needs its bounded review; no additional model inference is implied. The neighbor allocation screen remains +.003 over both contemporary controls and retained .44716940136585204. No automatic extra seeds, extensions, averaging or TEST release.

Current action: hold all scientific execution. Keep sealed sources/reports unchanged and these mutable handoffs stable during QC's publication transaction. Existing four-hour monitoring is unchanged. Latest completed publication remains Round07 at703c2cd7 until the publisher verifies the new receipt.
