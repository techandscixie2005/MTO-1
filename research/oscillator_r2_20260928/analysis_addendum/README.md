# Validation comparison addendum

This directory contains a CPU-only, validation-only analysis of the frozen MTO loss-pilot and architecture screens. It does not train, change checkpoint selection, inspect running checkpoints, or read historical test predictions.

## Run commands on USTC-A800

From `/home/inspur/MTO-1/research/oscillator_r2_20260928/analysis_addendum`:

```bash
PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
$PY validation_comparison_addendum.py --study fixtures
$PY validation_comparison_addendum.py --study pilot
$PY validation_comparison_addendum.py --study architecture
$PY validation_comparison_addendum.py --study all
```

Run `fixtures` now. The study commands return a machine-readable `pending` response until **all** arms have `FIT_COMPLETE.json`, full history, selected validation predictions, and the original round summary. They then write `PILOT_VALIDATION_ADDENDUM.{json,md}`, `ARCHITECTURE_VALIDATION_ADDENDUM.{json,md}`, and, for `all`, `COMMON_COMPARISON.{json,md}`. Pending checks inspect only file existence. Do not use this script as a routine job monitor.

## Method and checks

- Truth is the original printed raw oscillator strength `f` on the frozen 6,686-molecule validation split. Each molecule has ten states. Arrays must match the frozen indices, unique molecule IDs, exact raw labels, masks, and source hashes. The selected prediction alias must hash-identically to the selected epoch artifact; the selected checkpoint hash and epoch are recorded.
- The primary metric is pooled native-`f` R²: `1 − SSE / pooled SST`. Epoch 0, the validation-selected lowest-SSE checkpoint, and fixed epoch 20 are reported with raw-`f` R²/MAE/RMSE and energy MAE. The JSON also retains the full 21-point validation R² curve. A-derived truth is reported separately as sensitivity, never substituted for raw `f`.
- Paired bootstrap resamples either molecules or the original conservative RDKit connectivity groups, with all ten states and every molecule in a sampled group kept together. It recomputes pooled SST on each of 2,000 replicates (seed 20260928). Group mapping comes only from `identity_audit_v2.json` at validation indices.
- State `j` contributes `(SSE_control,j − SSE_arm,j) / pooled SST` to the primary ΔR². The ten contributions must sum to pooled ΔR². Fixed q90/q99 thresholds come from the same raw validation labels for both rounds. Bright and complementary subsets report counts, SSE, delta, and relative SSE reduction versus that round's matched control.
- Compare pilot arms against pilot/control and architecture variants against architecture/original. The pilot direct-`f` denominator is 0.011321718689897071; architecture uses train-only `Var(f)=0.002510981243894732`, making architecture `f` error 4.508882221810699 times as strong relative to the same energy term. Pilot control/weighted instead use trace losses. Architecture/original versus pilot/direct_f_matched is a loss-scale comparison, not an architecture gain.
- Distilled new-head architecture arms have their own train-only teacher initialization at epoch 0. Original and retained-residual start at the eta0 prediction; do not describe distilled heads as exact eta0 starts. All summaries describe selected **single** models. The earlier equal-three ensemble is a separate benchmark.

Intervals condition on each already selected checkpoint. They do not adjust for selection among 21 epochs or multiple arms, and one seed cannot measure training variability. The historical test was previously reused and is not opened here.

## Fixture verification

`FIXTURE_VALIDATION.json` uses the existing eta0/eta01/eta1 validation predictions only. It reproduces the published equal-three validation R² and verifies exact ID/index/raw-truth alignment, whole-molecule and connectivity-group resampling units, and the sum of per-state contributions. It is an analysis fixture, not a new experiment or test result. Only lightweight JSON/Markdown/code should be archived to D: and GitHub.

