# Round04 runner handoff — PREPARATION ONLY

The future fit/export/evaluation code is implemented and awaiting final source review/publication. **No real-data coefficient fit, outer-validation comparison or production launch has occurred.** Preparation publication will not authorize any of them. There is no production execution authorization file and no `production/` directory.

## Preserved phases

- `documents_only_snapshot/`: accepted four-coefficient protocol, rationale, manifest and independent protocol review before implementation.
- `stub_preparation_snapshot/`: original basis/wrapper/stub sources and their target-free/synthetic preflight records.
- Current sources: the same mathematical basis and wrapper, plus a complete future runner behind separate execution gates. `PREFLIGHT_CONTINUITY.json` proves the unchanged design/inference helpers, so the original input-design and model-fixture checks were not needlessly repeated after the runner was added.

Current preparation checks use two CPU threads and `CUDA_VISIBLE_DEVICES=` set before interpreter startup. They include synthetic least-squares recovery and boundary/failure cases; a target-free audit of the two frozen calibration prediction sources; synthetic-geometry one-checkpoint parity using unchanged full eta0; and synthetic authorization/recovery/refusal tests. No GPU allocation is used.

## What a future authorized run does

`production_entry.py` verifies the separate execution decision, final source manifest, independent review and archive-first remote publication before importing `production_stages.py` or creating a stage artifact. The authorization must explicitly bind the exact hashes of those three files and permit exactly two real-data solves and one fixed validation comparison, with test access false. A preparation decision, synthetic test fixture or publication alone is rejected.

Both stages run synchronously under a nonblocking `production.lock`. The fitting stage retains all 240,710 valid calibration labels, solves each source once, and writes its four coefficients into a server-only FP64 tensor. Before each solve it fsyncs a STARTED marker. If a process stops after STARTED but before a coefficient tensor survives, recovery stops for review rather than automatically repeating the solve. Existing tensors recover missing receipts or exports without refitting. Tensor and JSON commits flush/fsync the file before rename and then fsync the directory.

After both tensors are immutable, the stage writes a coefficient-free hash receipt, exports both complete original eta0-plus-map checkpoints and performs the fixed first-64-calibration geometry replay. The scalar mapping and native-base comparisons use separate frozen tolerances. Both exports must preserve full-base tensors, original configuration/statistics and exact permit/arm/coefficient provenance.

Only after those gates may `evaluate` open the frozen validation cache. It applies both maps to the same saved full-baseline native f, with no new full-model validation forward. It records an attempt before accessing validation values, reports all six fixed predictors and complete pooled/state/tail/bin metrics, and applies the accepted promotion/selection rules. Any completed or prior-attempt marker refuses automatic repeated scoring. Interrupted validation needs a separate recovery decision; the current runner provides no bypass.

## Future commands — not currently authorized

All paths are on USTC-A800. The working directory is `/home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation`.

```bash
MTO_PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MTO_PY" production_entry.py fit_export \
  --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json \
  --manifest FROZEN_MANIFEST.json --review INDEPENDENT_PREPARATION_REVIEW.json \
  --publication ../ops/ROUND04_PREPARATION_PUBLICATION_RECEIPT.json

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MTO_PY" production_entry.py evaluate \
  --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json \
  --manifest FROZEN_MANIFEST.json --review INDEPENDENT_PREPARATION_REVIEW.json \
  --publication ../ops/ROUND04_PREPARATION_PUBLICATION_RECEIPT.json
```

Preparation PASS, publication and the root preparation closeout do not grant production execution. The saved heartbeat expressly prohibits a new fit. A later instruction must authorize that scope before root may issue the separate bound execution decision; future heartbeats otherwise remain monitoring-only. No preparation script creates an authorization. Preserve unique stage logs if execution is later authorized. There is no automatic seed launch.

Its exact location is `/home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json`. The required schema is shown below as documentation only; the placeholders are not valid authority and no file with these fields has been created:

```json
{
  "phase": "production_execution",
  "real_target_fit_authorized": true,
  "fixed_validation_authorized": true,
  "test_access_authorized": false,
  "source_manifest_sha256": "ROOT_MUST_BIND_THE_FINAL_MANIFEST_SHA256",
  "independent_review_sha256": "ROOT_MUST_BIND_THE_FINAL_REVIEW_SHA256",
  "publication_receipt_sha256": "ROOT_MUST_BIND_THE_VERIFIED_PUBLICATION_RECEIPT_SHA256",
  "maximum_real_data_solves": 2,
  "maximum_validation_comparisons": 1
}
```

The review must bind the same manifest and complete `source_hashes` dictionary. The publication receipt must contain `source_manifest_sha256`, `independent_review_sha256`, `remote_verified: true` and `download_before_stage_before_commit_push: true`. A changed source, binary-input hash, review or publication receipt rejects execution. Both future commands shown above are currently blocked; no claim of a ready execution authorization is made.

## Numerical preparation evidence and limitations

Both fixed calibration designs have rank four. Condition numbers are 10.3199979469 (in-sample) and 9.6517799555 (held-out), below the predeclared limit 1e8. Their source-input upper-knot counts are 1600 and 1252 respectively. This supports numerical identifiability of the fixed basis, not predictive accuracy or robust rare-tail estimates. The audit decoded five permitted input/mask/identity members and no target/energy member. The 4546 true-zero count comes from the pinned historical receipt.

Synthetic coefficient recovery error was 1.51e-16 and affine nesting error at most 2.22e-16. The one-checkpoint fixture preserved full eta0 state/buffers and E/A bitwise; nonlinear NumPy/Torch mapping differed by 1.3877787807814457e-17 and the affine fixture by zero. No external file was opened during the guarded load/forward. Synthetic fixtures are not accuracy evidence.

The scalar map cannot distinguish different transitions having the same native f. Its upper segment extrapolates linearly but may be steep or nonmonotone; the clamp prevents negative outputs, not large positive errors. The hypothesis follows historical validation diagnostics. Source data size/quality differ; no causal interpretation or fresh confirmation is justified. Mapped f may be inconsistent with unchanged E/A. Existing calibrated eta0 remains the incumbent until an authorized comparison is completed and reviewed.

## Archival boundary

Only lightweight sources/settings/logs/receipts, aggregate diagnostics, reviews and decisions are eligible for the preparation archive. No checkpoints, coefficient tensors/arrays, exact new learned slopes, raw labels, indices, prediction arrays or caches may be uploaded. Synthetic hand-set constants are labeled in test source. The two prior snapshots must be retained. Monitor first downloads and verifies records in `D:\MTO\archives\`, then inspects staged files, commits/pushes and verifies the remote. Root owns every later execution decision.
