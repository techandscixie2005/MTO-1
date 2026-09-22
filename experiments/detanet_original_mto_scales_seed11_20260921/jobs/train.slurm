#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --job-name=MTO11_AB
#SBATCH --array=0-1%2
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921/logs/train-%A_%a.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$PWD"
PY=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/env/bin/python
variants=(detanet_original_uv detanet_mto_planned)
"$PY" -u train.py --scale "$1" --seed 11 --variant "${variants[$SLURM_ARRAY_TASK_ID]}" &
child=$!
checkpoint_and_requeue() {
    trap '' USR1
    kill -USR1 "$child" 2>/dev/null || true
    set +e
    wait "$child"
    result=$?
    if [ "$result" -eq 75 ]; then
        task="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        scontrol requeue "$task"
        rq=$?
        if [ "$rq" -ne 0 ]; then exit 75; fi
        exit 0
    fi
    exit "$result"
}
trap checkpoint_and_requeue USR1
wait "$child"
