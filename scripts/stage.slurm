#!/bin/bash
#SBATCH --job-name=MTO_fullarch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/run01/sczc698/xxy/MTO_fullarch_20260914/logs/stage-%j.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO_fullarch_20260914
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
PY=/data/run01/sczc698/.conda/envs/cheng/bin/python
"$PY" -u scripts/stage_runner.py --scale "$1" &
CHILD=$!
trap 'kill -USR1 "$CHILD"; wait "$CHILD"; exit 75' USR1
wait "$CHILD"
