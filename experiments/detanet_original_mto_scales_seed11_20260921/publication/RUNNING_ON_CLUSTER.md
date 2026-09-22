# Single-seed DetaNet / MTO scale experiment

Workspace: `/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921`.

This independent run uses seed **11**, exactly two models, and three existing data scales. It does not incorporate historical metrics or checkpoints. The complete DetaNet implementation and vendor source come from `../detanet_original_mto_1k_20260920` at commit `15af391115fb5179521397ea7f75b59153dac482`. Source copies and hashes are in `audit/`. The only model construction change is to omit the global-gate branch; A/B backbone initialization is unchanged. Architecture: 128 features, maxl=3, 3 blocks, 32 trainable Bessel radial bases, 8 attention heads. A retains the original direct 240-bin spectrum head; B uses the existing planned MTO readout.

| Scale | Train | Validation | Test |
|---|---:|---:|---:|
| 1k | 800 | 100 | 100 |
| 10k | 8000 | 1000 | 1000 |
| full | 103785 | 12973 | 12974 |

The source is the existing full valid 129732-molecule NPZ and the three exact split JSON files under `/data/run01/sczc698/xxy/MTO_fullarch_20260914/data`. The preparation job checks the source hash, molecule identities, geometry, finite nonnegative spectra, graph validity, exact split membership, and scale sizes. All scales use `numpy.interp` from 601 source points onto the author FP32 `torch.linspace(1.5,13.5,240)` coordinates, with FP64 interpolation and FP32 training targets. A single RMS and median atom count are calculated only from each scale's training set. The source NPZ stores FP32 spectra, unlike the old 1k JSON's numeric text representation; the interpolation rule is unchanged and both arms share exactly the same prepared targets. Source broadening and absolute physical intensity units remain unverified; MTO retains sigma=0.2 eV.

Training retains Adam + AMSGrad, lr=1e-3, weight decay=0, effective batch=64, FP32 without AMP/TF32. ReduceLROnPlateau: factor=.5, patience=50, relative threshold=1e-4, minimum LR=1e-6. Budget is 1000 epochs; normal early stopping requires at least 200 epochs, 150 epochs without improvement, at least two actual LR reductions, and 50 epochs since the last reduction. Each scale starts fresh; each arm selects its own best validation checkpoint. Shuffle seed is 1000014 for both arms. No performance gate, hyperparameter search, additional seeds, global-gate training, or diagnostic experiment is run.

Before a fresh fit, two training molecules undergo a short finite forward/backward check; no optimizer update or check checkpoint is retained. The checked model is discarded and the formal model is freshly constructed. `short_check.json` records the real outcome. There is no `PREFLIGHT_PASSED` file. Nonfinite training losses, gradients or validation losses fail the job. Forward/backward OOM halves micro-batch size and retries the same effective batch with sample-weighted gradient accumulation; changes are recorded. Optimizer-step OOM is allowed to fail rather than retry a potentially partial optimizer update.

## Submission and ordering

```bash
ssh bjhpc
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921
PY=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/env/bin/python
PYTHONUTF8=1 SBATCH_EXPORT=ALL,PYTHONUTF8=1 "$PY" submit.py
```

The Slurm chain is `prepare -> 1k train[0-1] -> 1k report -> 10k train[0-1] -> 10k report -> full train[0-1] -> full report`. Every arrow is `afterok`. Array task 0 is A, task 1 is B, each with one GPU and array concurrency two. Preparation and reports each request one GPU in the existing `gpu` partition. There is no overlap between scales. Training independently verifies the preceding scale's artifact hashes and completion. Reports require exactly two actual completed fits, evaluate only their best validation checkpoints, then verify all required output files before writing `STAGE_COMPLETE.json`. No accuracy threshold controls advancement. The full report also writes `reports/summary.md`, `summary.json`, and `WORKFLOW_COMPLETE.json` after verifying all three stages.

`submit.py` uses a file lock and `jobs/submission.json` to reuse active jobs and avoid duplicate submission. Re-running it skips completed, verified work, resubmits failed/missing nodes, and updates this workflow's pending downstream dependencies. It never cancels other jobs. Do not change frozen Python/config/vendor/Slurm files once a fit has started: checkpoint fingerprints reject changes. README and submission receipts can be updated without changing training fingerprints.

## Status and recovery

```bash
squeue -u sczc698
cat jobs/submission.json
tail -n 20 logs/train-<ARRAY_JOB_ID>_0.log
tail -n 20 logs/train-<ARRAY_JOB_ID>_1.log
cat reports/1k/comparison.md
PYTHONUTF8=1 SBATCH_EXPORT=ALL,PYTHONUTF8=1 "$PY" submit.py  # idempotent retry/resume
```

Use `sacct -j <job_ids> --format=JobID,JobName,State,Elapsed,ExitCode` for completed or failed jobs. `last.pt` contains the model, optimizer, scheduler, all RNG states, epoch, shuffle order, batch cursor, elapsed time and history. It is saved every 50 updates, at epoch boundaries, and on graceful interruption. The Slurm 24-hour training allocation sends USR1 five minutes before timeout; the wrapper requests a checkpoint and automatically requeues the same array task on exit code 75. A failed requeue or hard kill does not count as success and does not release the report dependency; rerun `submit.py` to resume from this run's last checkpoint. A time limit never produces `FIT_COMPLETE.json` unless the configured training budget or normal early-stop condition was genuinely reached. Reports can be rerun from completed fits without retraining.

The cluster batch locale defaults to ASCII. `PYTHONUTF8=1` must be exported into report jobs so Markdown characters such as superscript 2 can be written. The initial 1k report failed at Markdown writing after test metrics were saved; the operational UTF-8 repair and preserved partial artifacts are recorded under `audit/utf8_recovery_20260921/`. Scientific code, checkpoints and training fingerprints were not changed.

## Outputs

- `data/frozen_<scale>.json`: actual counts, train RMS, n_ref, grid, split and target hashes.
- `runs/<scale>/seed_11/<variant>/`: `short_check.json`, `best.pt`, `last.pt`, `history.json`, optional OOM record, and real `FIT_COMPLETE.json`.
- `logs/`: Slurm preparation, A/B training, and report logs.
- `reports/<scale>/`: raw and train-RMS normalized MSE, MAE, cosine, training seconds/epochs/best epoch, CSV/Markdown comparison, training/validation/LR curves, test prediction NPZs, and overlays of the first four fixed test IDs.
- `reports/summary.md`: final three-scale comparison with MTO relative MSE improvement `100*(A-B)/A` and cumulative training GPU hours. Negative improvement means MTO is worse.

Single-seed preliminary comparison only; no across-seed standard deviations or significance conclusions. Existing historical split identities are reused and are not represented as untouched test sets. GPU training wall time excludes queue, preparation and short-check time.
