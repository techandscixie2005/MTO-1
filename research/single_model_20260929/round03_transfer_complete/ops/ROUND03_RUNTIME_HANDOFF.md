# COMPLETE — Round03 source training and affine comparison

This operational handoff supersedes every earlier pending-stage instruction. Source training, coefficient fitting/export and the single validation comparison are COMPLETE. Do not launch a source, refit coefficients or repeat validation from this handoff. No GPU job remains owned by science. This heartbeat authorizes no new model preparation or fit.

## Ownership and next action

Root owns the persisted Round03 decision and goal lifecycle. History owns the independent terminal/scientific review. Monitor owns the required completion archive: FIRST download lightweight records to D:\MTO\archives\, verify and inspect them, THEN commit/push and verify the remote. Science has finished implementation, execution, aggregate analysis and reproduction documentation. The four-hour heartbeat must read the current coordinator and decision before any future work; no source/affine continuation is pending.

Server root: /home/inspur/MTO-1/research/single_model_20260929
Round directory: round03_transfer
Pinned Python: /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python

## Terminal evidence

- Original source PID 1174422, registry round03_source33_1790707949921845597, completed exactly 33 epochs and 49,665 optimizer updates; the registered process exited. No source checkpoint selection used calibration, outer validation or test labels.
- Source FIT_COMPLETE, last.pt and source_final.pt agree. All 33 training orders, all 69 frozen hashes, initialization, fit-only statistics and split contracts passed terminal review. Total complete-epoch compute: 3921.72 seconds.
- Coefficients were frozen before the sole outer-validation attempt. Both deployment exports contain the same complete original full baseline and one coefficient pair. There is no model/checkpoint averaging or auxiliary source at inference.
- affine/VALIDATION_COMPLETE.json exists; all 66,860 valid labels, including zeros, were scored. No test inference occurred. Selected source is fixed epoch 33; there was no validation checkpoint search in this round.
- INDEPENDENT_TERMINAL_INTEGRITY.json and INDEPENDENT_SCIENTIFIC_REVIEW.json record PASS. Independent review manifests bind their inputs. No further model inference is needed for closeout.

## Result and decision

Native validation pooled raw-f R²: 0.405294124.
In-sample affine: 0.402024351.
Held-out-source affine: 0.416374515.
Historical validation-fitted affine incumbent: 0.418119244.

Held-out transfer beats native by 0.011080390 and in-sample transfer by 0.014350164, passing both prespecified 0.003 preparation thresholds. It trails the incumbent by 0.001744729. Retain the calibrated eta0 incumbent; no new-best or independent-seed promotion follows. Eight state R² values improve relative to native; S1 and S5 worsen. MAE and true q90/q99 tail RMSE also worsen relative to native. Historical validation exposure remains; no fresh holdout confirmation is claimed.

Root's ROUND03_DECISION.md records a future proposal for matched f-only nonlinear readouts with same-input affine controls. No such implementation or fit is authorized in this heartbeat. Scratch F/decorrelation and AO fitting remain deferred.

## Checkpoints — server only

Under round03_transfer:
- runs/source33/source_final.pt SHA256 769026edf4e999799dd201c3e8f2ad3393e25ef9b3a5a99b23f7ce59ffc984ec
- runs/source33/last.pt SHA256 52f2884b4869ae87d85ba56875021ff3d42383cd53bfad356362d213883499c9
- affine/heldout_source.pt SHA256 291b80c87afc0c841781d3c7e1b376f6008a13d955291110d5305280d75df398
- affine/in_sample.pt SHA256 3f618f960492af279c6d569ee5b79cf1b43af86ba48d5d4fd53b38cb66d63387

Incumbent baselines/calibrated_eta0.pt SHA256 bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9.

Coefficient receipt affine/COEFFICIENTS_FROZEN.json SHA256 cb1c355304033fed9a00f759925b84ded8e0b31ead4639652e68f943d1d54ce6.
Export receipt affine/EXPORT_COMPLETE.json SHA256 26d766b668035f83e533a5899381bedba7af0635a9ee7bc7f3a4ea0c18fac065.

## Device placement and reproduction

The registered source ran on physical GPU1. During the short affine-fit interval, monitor observed MTO Python PID 1186535 allocating 504 MiB on GPU0. It exited before argv/start identity could be captured. The fit's exact full placement cannot be established; do not claim it ran wholly on GPU1 or that GPU0 was unused. The completed fit and coefficients were preserved without repetition. No process was signaled, no unrelated job was modified, and performance impact was not measured.

Validation was launched once with CUDA_VISIBLE_DEVICES=GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800 in the child environment BEFORE Python startup. Owned PID 1187988 was positively observed on that GPU1 UUID and exited with code 0. Physical GPU1 maps to logical CUDA device 0; --gpu 1 remains the physical admission/lock index. Future GPU invocations must apply this prebinding before imports. The existing resource health checks/shared lock remain mandatory. SOURCE FILES AND COEFFICIENTS WERE NOT CHANGED.

Exact evidence: RESOURCE_PLACEMENT_NOTE.md, GPU_PLACEMENT_REVIEW.md, VALIDATION_LAUNCH_20260929T200405597244Z.json, monitoring/ROUND03_HEARTBEAT_20260929T195916.json. Full commands and geometry-only one-checkpoint loader usage are in ROUND03_REPRODUCE.md. They document reproduction; they do not authorize rerunning completed stages.

## Final records and publication

Under round03_transfer:
- ROUND03_REPORT.md SHA256 ed36611de9f9945f55838f209781d5add7a07212cc6652f347e610f30f2037b2
- ROUND03_REPRODUCE.md SHA256 c29a4052090e7c322aae6e06673c6e309c50b2db45bead7ae1f93afe7c1fa247
- ROUND03_RESULTS.json, ROUND03_RESULTS.md, ANALYSIS_RECEIPT.json
- verify_source_gate.py, TERMINAL_SOURCE_GATE.json
- summarize_transfer.py, SUMMARY_EXECUTION.log
- run_validation_bound.py, the unique fit/evaluation logs and validation-launch receipt
- Independent integrity/scientific reviews, root ROUND03_DECISION.md and complete source/coefficient/export/validation receipt records

Monitor's ops/ROUND03_COMPLETION_ARCHIVE_NOTES.md also includes supplemental next-direction, same-input control and AO feasibility reviews. The final report/reproduction documents are frozen; operational handoffs may be updated after verified publication without changing sealed preparation snapshots.

Preparation commit: 1b34e839335177e1fd582944549be4f181dee3c0, descending Round02 completion bdc28c351b941a58dae49f833284e13cf5fdc46b. Preparation manifest SHA256 93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37. Completion publication is owned by monitor and pending in this handoff. All checkpoints, optimizer states, datasets, indices, calibration/validation arrays and caches remain server-only.
