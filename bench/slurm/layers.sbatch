#!/bin/bash
# M2 per-layer bench (3_layers.md), one GPU. Usage:
#   sbatch -p priority --time=00:30:00 -o measured/logs/%j.out bench/slurm/layers.sbatch <lengths> [extra args]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-layers
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppm$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} lengths=$1 args=${*:2}"
uptime
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
$PY -m bench.layers --lengths "$1" "${@:2}"
uptime
rm -rf "$TMPDIR"
