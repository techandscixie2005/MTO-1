# Independent review handoff — Round05 completed

All four authorized fresh seed11 arms completed60epochs/112860updates with one original attempt each, no failure/resume, unchanged82-file source66269e04.... No TEST scoring or repeated model inference occurred in closeout.

## Verified results

| Arm | Selected epoch | Validation pooled raw-f R2 | Fixed60 R2 |
|---|---:|---:|---:|
| control |45|.4471694014|.4026919323|
| adapter |41|.4150203688|.4011779433|
| decorrelation |23|.4123991927|.3920134759|
| both |20|.4249660535|.4023910909|

All66860 labels retained. No modified arm passes+.003 versuscontrol. Review recommends close the bounded factorial, no continuation/confirmation for these variants. R2.60 remains unmet. New v2scores are not directly comparable to the historical split's incumbent. The v2TEST remains historically exposed data, not fresh external confirmation.

Current v2 reference: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e. One model/checkpoint with embedded newTRAIN stats/config, geometry-only inference. No binary is published.

## Review records

Server namespace: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation`; local `research_state/round05_scratch_preparation`.
- INDEPENDENT_TERMINAL_INTEGRITY.json417950cf...:82sourcepins,61histories/arm,60orders,steps,selection,coverage/access and opaquecheckpoint hashes.
- TERMINAL_ANALYSIS_SOURCE_REVIEW.jsonb1ffea47...: reviewed CPU saved-output analysis, no model construction/rawtarget reads.
- completion/ANALYSIS_RECEIPT.json31bc9468... and ROUND05_RESULTS.jsonbf6920dd...: science checkpoint/export tensor checks and eight saved validation output reductions.
- completion/INDEPENDENT_RESULT_CHECKS.json8ebe5538...:54 source/input/output hashes,history/result arithmetic,gate/interaction/trajectory reconciled.
- completion/INDEPENDENT_TERMINAL_REVIEW.jsonf6b5d637f7fe7128664c9d9e22e8d1d3e2aea8d9425b35c2f6d632fb230c44dc; scientific Markdown0e53e364....

F is active and raw-M penalty reduces overlap; this does not establish physical state orthogonality/phase covariance. TRAIN diagnostics are trajectory aggregates. Selected F improves tails but loses pooled/false-bright errors; no uniform-degradation claim. Component bootstrap is descriptive conditional checkpoint-selected reused-validation uncertainty, not seed or fresh-generalization confidence.

## Pending closeout

Root accepted the completed negative factorial and closed it without extension/seeds. Final science report and reproduction guide are sealed. Publication-only completion/INDEPENDENT_CLOSEOUT_MANIFEST.json SHA bf6007f6f1ffdc4b08bda28a453e0e842c7610b3e642d1793a6fdafc1dfe01d8 binds 13 reviewed records and root decision. All member hashes and 52 science terminal-manifest records were verified. QC performs D-first archive then exact staged byte review and non-force publication. README candidate SHA 4eb3743b051a3e3b1cdb2f07d9aef6c64c71817f2641112388fcb3e8484ce82c passed scientific review; root exact approval and publication checks remain. Do not repeat completed fits/preflights/analysis or open TEST. All arrays/checkpoints remain private; entire private_partition/private_preflight* excluded.

Current main/research preparation publication:9de80e37e1586c2d3068b188ed2c242ee9883147. Next publication descends from this verified tip. No new experiment or extension is authorized by this handoff. Preserve completed archives and frozen sources.
