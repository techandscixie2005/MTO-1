# Round07 preparation accepted; publication pending; production blocked

Root accepted preparation in ROUND07_PREPARATION_ACCEPTANCE.md, SHA89ffc0d7d04b99eecafdebdf9cf54fea2a89b0522ec8fd1e2bf1a20dfe704826. QC owns its mirror, the current coordinator/monitor and D-first publication. Science owns this operational handoff and remains sole future executor. Sealed sources, reports and prior rounds remain unchanged.

## Completed evidence — do not rerun

CPU synthetic and nine-update GPU stages are COMPLETE and independently reviewed. Exact scope: first128 TRAIN rows/two64 batches, update1/update2/replayed update2 per original/scalar/tensor arm, nine discarded updates total. No production run or validation/TEST numeric access occurred. Fixture states remain private and must never initialize production.

- Frozen185-file manifest:68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203.
- Final preparation review:e3ae0804141dc37fdbcb3797411a301e75dabe223eea4005d0c043197f423826.
- Supplemental review manifest:459d1a35dc2f39ab02f64b600c0731c391cd633584112e25b94bfe53d49b4687.
- Implemented protocol:ee22c436332e753de1636e3db5e6587a628f6e7925965748cd1a626abf710bda.
- Report:27e1e0c72bfcdceef11c3c55b07548d20b1e39d9b43ad8e04f026c883312350c.
- GPU receipt:24e891e92c8c670361f406363153c5a39719106416108bcfc9b56813598296b6; independent GPU review:e0a882ff6d44c9f221dbda56d46510a99dccf0105e2d3efa9afb1ff60b9a1b5b.

The registered preflight child1547487/start1271943701 exited0 on physicalGPU1 and released its shared lock. Metadata freeze ran once, log ops/freeze_preparation_01.log. These are engineering checks, not validation accuracy or identified QC physics.

## Exact remaining authority gates

FIRST download lightweight preparation records to D:\MTO\archives, inspect exact staged bytes, commit/non-force push and verify remote publication. Expected receipt: /home/inspur/MTO-1/research/single_model_20260929/ops/ROUND07_PREPARATION_PUBLICATION_RECEIPT.json. It must bind the manifest and review above and assert remote_verified plus download_before_stage_before_commit_push.

Root must then issue a separate /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json. Required schema:

- authorized:true; scope:round07_three_arm_60epoch_fit.
- frozen_manifest_sha256:68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203.
- independent_review_sha256:e3ae0804141dc37fdbcb3797411a301e75dabe223eea4005d0c043197f423826.
- publication_receipt:the exact server receipt path above; publication_receipt_sha256:the verified receipt's actual SHA after publication.
- split_manifest_sha256:c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155.
- independent_split_verification_sha256:395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2.
- epochs:60; ordered arms:[original,scalar,tensor]; test_access:false; historical_weights:false.

Do not create that authority from this handoff. The frozen false template, root preparation acceptance, publication and monitoring heartbeat do not themselves authorize production. There is no production authority or run directory at the final reviewed seal. Recheck actual state before any future launch rather than assuming this snapshot remains current.

## Future command — blocked until both gates exist

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /usr/bin/python3 launch.py \
  --authorization /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json \
  --arms original scalar tensor
```

The reviewed launcher verifies exact authority/source/review/publication before mutation, then sequentially admits original/scalar/tensor on physicalGPU1/2/4 under separate exclusive shared locks. Children run concurrently. UUIDs are prebound before interpreter imports; logical CUDA0 maps to each assigned physicalGPU. Each attempt records owned PID/start/boot/UID/cwd/argv, source/authority and actual registration. No fallback GPU, extra seed, budget extension or TEST evaluation.

For admission failure, partial launch or child failure, preserve each attempt/log/status/checkpoint and inspect exact identities. Do not rerun all arms blindly or signal unknown processes. Completed markers and live worker locks reject relaunch. Only an explicit recovery decision may resume an incomplete arm with --arms <arm> and the unchanged bindings. last.pt is this runner's committed epoch resume entry with optimizer/RNG/order/history; mid-epoch recovery restarts from the previous completed epoch. best.pt is the selected snapshot, geometry_best.pt the standalone export. No fixture or prior trained state may be reused.

Existing four-hour monitoring remains unchanged. Notify QC of actual launch identities after admission; science performs first committed-epoch metadata review only after a genuine production launch. Current task requires no further numerical check or resource polling. Retain Round05 control45 reference R² .44716940136585204; TEST sealed; .60 unachieved.
