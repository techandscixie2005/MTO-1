#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --job-name=MTO11_prepare
#SBATCH --output=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921/logs/prepare-%j.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$PWD"
PY=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/env/bin/python
"$PY" -u environment.py
"$PY" -u data_protocol.py
