# Round03 durable continuation

## Current authority and scope

The coordinator conditionally authorized one original-MTO source trained on96,284 internal TRAIN molecules for exactly33 epochs, followed by the two fixed affine fits and one fixed outer-validation comparison. No nonlinear fitting, source-checkpoint search or test scoring is authorized. Root authorization is in round03_transfer/ROUND03_LAUNCH_AUTHORIZATION.md. Round02 completed publication is bdc28c351b941a58dae49f833284e13cf5fdc46b.

Server directory: `/home/inspur/MTO-1/research/single_model_20260929/round03_transfer`.
Python: `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`.
Frozen manifest SHA:93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37.
Independent review SHA:a6807aae55c59f254f2048b1d696a754f130075e32ef4763dfa1c94f4fb1c04c.
Both checks passed; publication receipt is the remaining launch gate as of this note.

## Source execution

The sealed launcher starts only train_source.py. It does not chain the affine stage. Once monitor publishes the manifest/review-bound receipt, the command from the server directory is:

```
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python launch_round.py --review INDEPENDENT_PRELAUNCH_REVIEW.json --publication-receipt ../ops/ROUND03_PUBLICATION_RECEIPT.json
```

It verifies all69 hashes, the exact review and archive-first remote publication, admits healthy free GPU1 under `/tmp/mto_pouter_gpu_1.lock`, launches and registers the owned job. Canonical launch metadata is runs/source33/LAUNCH_RECEIPT.json. Source progress is runs/source33/status.json, history.jsonl and train.log. Source failures are FAILED.json; source completion is FIT_COMPLETE.json. Atomic last.pt includes model, optimizer and RNG. Partial epochs replay from the last complete epoch on an authorized restart. Do not infer failure from an agent interruption and do not touch unrelated jobs.

## Completion gate and exact next steps

Science owns continuation during active work. The root's persistent four-hour heartbeat may dispatch the same steps if this turn yields; the server cron itself is read-only and does not launch the affine stage. Before running, inspect completion/failure/launch metadata and ensure the source process has exited and the shared GPU lock is available. FIT_COMPLETE must report epoch33 and the exact manifest; source_final.pt and last.pt hashes must match its receipt. Both reviewed stage commands revalidate the source manifest and final checkpoint, and independently admit/lock GPU1.

1. If affine/COEFFICIENTS_FROZEN.json or affine/EXPORT_COMPLETE.json is absent, run:

```
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py fit --gpu 1
```

Capture stdout/stderr as a uniquely named AFFINE_FIT_EXECUTION_<attempt>.log. If the coefficient receipt already exists after an interrupted export, the same command verifies its immutable prediction-array SHA and completes/rechecks exports without refitting. It never reads outer-validation targets. It writes both coefficients before exporting the original full baseline plus each pair.

2. When the frozen coefficient and export receipts exist and affine/VALIDATION_COMPLETE.json is absent, run:

```
/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python affine_stage.py evaluate --gpu 1
```

Capture stdout/stderr as a uniquely named AFFINE_VALIDATION_EXECUTION_<attempt>.log. This verifies both pinned exports, records an attempt, and evaluates all four fixed predictors from one original full-baseline native forward pass. It does not fit coefficients or use the auxiliary source for deployment. A completed result makes reruns refuse. An interrupted attempt can repeat the same frozen computation; preserve the original log/attempt receipt and do not change coefficients or settings. Do not run stages concurrently or bypass a busy/unhealthy GPU admission.

3. After VALIDATION_COMPLETE exists, science summarizes source history, source quality, both coefficient pairs, all-label/state/tail metrics, export/checkpoint provenance and limitations. History performs independent terminal review. Root records the next decision. Monitor FIRST downloads all lightweight records to D:\MTO\archives\, then inspects/stages/commits/pushes them. All checkpoints, optimizer states, index arrays, calibration/validation arrays and raw data remain server-only.

## Decision boundary

Heldout-source map must beat BOTH the in-sample map and native anchor by at least0.003 pooled validation raw-f R² to allocate nonlinear-head preparation. Per-state and bright-tail tradeoffs are retained. The historical validation-fitted map is an exposed comparator; merely approaching it is neither a new-best claim nor an independent-seed promotion. No automatic next fit follows this result.
