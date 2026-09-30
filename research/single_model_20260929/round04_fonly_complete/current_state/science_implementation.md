# COMPLETE — Round04 production and reporting

Both scientific stages completed normally on 2026-09-30. The later user interruption requested publication and did not interrupt either stage. No fitting, inference, validation comparison or recovery remains pending. Do not remove terminal markers or rerun completed stages.

## Current user priority

Push honest lightweight results to `main` and update repository README.md. Root/monitor own Git/archive publication, preserving existing uncommitted work. First archive the reviewed completed records to `D:\MTO\archives\`, then inspect staged files and commit/push. No model weights, learned coefficient arrays, raw data, prediction arrays, indices or caches may be published. Completed sources and prior records remain unchanged.

## Terminal evidence

- Server campaign: `/home/inspur/MTO-1/research/single_model_20260929`.
- Round namespace: `round04_fonly_preparation`.
- Fit/export attempt: `ops/execution/round04_fit_export_1790738868203678017`; exit 0, 14.40 seconds, registered CPU wrapper before child PID 1250296.
- Evaluation attempt: `ops/execution/round04_evaluate_1790738935645961587`; exit 0, 4.79 seconds. Exact wrapper/child identities, selected environment, authorization binding, logs and output hashes are in each LAUNCH_RECEIPT.json and COMPLETE.json.
- Both environments had CUDA_VISIBLE_DEVICES empty and OMP_NUM_THREADS=MKL_NUM_THREADS=2 before imports. Both terminal receipts verify authorization unchanged.
- Exactly two coefficient solves and one prescribed legacy-validation comparison completed. `production/VALIDATION_ATTEMPT_1790738939451324153.json` is the single scoring attempt. Scientific source manifest stayed `79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f`.
- Authorization SHA: `530db555da8badefb773b945e16f5e5ccce9c54fc796f78ec0d6a420b9dc6b97`.
- `ROUND04_REPORT.md` SHA: `4cdeafe40f86145b9020debf2d76d32a0771ddb7e914adc73a2df16e96a8e010`.
- `ROUND04_TERMINAL_MANIFEST.json` SHA: `8c5defec95b7eb2d06e16cb960b5d82f8981aae9ec153e2519dfc47ef8b3a396`.
- `report_completed.py` SHA: `26aef57e021daede1e477b6d530c473ad490045a7876eb0a491e0cd5ee2bd380`. This reads aggregate receipts only; an initial report-only filename error was corrected to the actual unique validation-attempt name. No production stage failed or repeated.

## Outcome

Pooled raw-f validation R²: native 0.405294124; incumbent historical affine 0.418119244; in-sample affine 0.402024351; held-out-source affine 0.416374515; in-sample fixed hinge 0.398331068; held-out-source fixed hinge 0.411775449. Both new maps lose against their matched affine controls and fail confirmation eligibility. The source contrast remains positive, but scalar flexibility did not help. Retain the incumbent; no spline continuation or seed allocation.

All 66,860 historically reused validation labels were included. All states/tails/brightness bins are in the report/receipt. No old test or new test was accessed. These scores cannot substantiate the requested 0.60 on a new benchmark. Rank/conditioning and synthetic parity are readiness checks, not accuracy gains.

The two private one-checkpoint exports are `production/in_sample.pt` (SHA f43897f1bc23b2981d617f3f0bd11fbc84499f6c2d162eb13227934ef989981f) and `production/heldout_source.pt` (SHA 5330d4028f460043883c08f66c4653cf75f0cb5a82a8716c7554b8f2b15619af). The report includes the unchanged incumbent location/hash. All remain server-only.

## Pending coordination, no new execution

History owns independent terminal review of the exact report/manifest and root writes the decision. Monitor then handles archive-first publication. Root owns the main-branch/README integration requested by the user. Four-hour monitoring persists without repeating completed jobs.

New split protocol received a preliminary favorable source/protocol review only. No split arrays, model preparation or training were performed during this publication task. Any new benchmark requires fresh initialization and new TRAIN-only statistics; historical checkpoints/calibrations are ineligible. The qc_design agent owns the bounded structural-options review. Detailed next protocol remains a later root decision and independent review; no new model launch is authorized by this handoff.
