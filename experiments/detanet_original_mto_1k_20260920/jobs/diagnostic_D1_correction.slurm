#!/bin/bash
#SBATCH --job-name=DetaOrig_D1_DIAGNOSTIC
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/logs/D1-correction-%j.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$PWD"
PY="$PWD/env/bin/python"
"$PY" -u environment.py > "logs/environment-${SLURM_JOB_ID}.json"
"$PY" -u tests/test_smoke_diagnostics.py > "logs/diagnostic_contracts-${SLURM_JOB_ID}.log" 2>&1
cp logs/diagnostic_contracts.json "logs/diagnostic_contracts-${SLURM_JOB_ID}.json"
"$PY" -u audit/rerun_D1_isolated.py
