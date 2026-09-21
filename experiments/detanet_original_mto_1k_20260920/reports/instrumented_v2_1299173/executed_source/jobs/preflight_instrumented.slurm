#!/bin/bash
#SBATCH --job-name=DetaOrig_MTO_1k_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/logs/instrumented-%j.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$PWD"
PY="$PWD/env/bin/python"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "logs/gpu-${SLURM_JOB_ID}.csv"
"$PY" -u environment.py > "logs/environment-${SLURM_JOB_ID}.json"
"$PY" -u tests/test_smoke_diagnostics.py > logs/diagnostic_contracts.log 2>&1
"$PY" -u tests/acceptance.py --device cpu --dtype float64 > logs/acceptance_cpu_float64.log 2>&1
"$PY" -u tests/acceptance.py --device cuda --dtype float64 > logs/acceptance_cuda_float64.log 2>&1
"$PY" -u tests/acceptance.py --device cuda --dtype float32 > logs/acceptance_cuda_float32.log 2>&1
"$PY" -u preflight.py
