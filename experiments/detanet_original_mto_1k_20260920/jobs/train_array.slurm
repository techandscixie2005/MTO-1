#!/bin/bash
#SBATCH --job-name=DetaOrig_MTO_1k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-14%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920/logs/train-%A_%a.log
set -euo pipefail
cd /data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$PWD"
seeds=(11 23 37 53 71)
variants=(detanet_original_uv detanet_mto_planned detanet_mto_global_gate)
index="$SLURM_ARRAY_TASK_ID"
"$PWD/env/bin/python" -u train.py --seed "${seeds[index / 3]}" --variant "${variants[index % 3]}" &
child=$!
resume_at_time_limit() {
    kill -USR1 "$child"
    set +e
    wait "$child"
    result=$?
    if [ "$result" -eq 75 ]; then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    exit "$result"
}
trap resume_at_time_limit USR1
wait "$child"
