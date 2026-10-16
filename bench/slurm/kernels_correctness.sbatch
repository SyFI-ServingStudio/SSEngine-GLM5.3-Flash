#!/bin/bash
# M1 correctness (2_kernels.md §5). Usage:
#   sbatch --time=00:25:00 -o measured/logs/%j.out bench/slurm/kernels_correctness.sbatch <rows> <configs> [kda-flashinfer]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-kcorr
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
ROWS=${1:-smallest,largest}
CONFIGS=${2:-}
OUT=measured/kernels_correctness.jsonl
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} rows=$ROWS configs=$CONFIGS"
uptime
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv
if [ "${3:-}" = "kda-flashinfer" ]; then
  PY=/raid/kanzhu/ServingStudio/profile_envs/flashinfer_kda/bin/python
  $PY -m bench.kernels_correctness --out "$OUT" --rows "$ROWS" --configs 5 --kda-flashinfer
else
  PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
  export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
  $PY -m bench.kernels_correctness --out "$OUT" --rows "$ROWS" --configs "$CONFIGS"
fi
uptime
