#!/bin/bash
# M2 per-layer bench (3_layers.md), one GPU. Usage, from the repo root:
#   sbatch -p priority --time=00:30:00 -o measured/logs/%j.out bench/slurm/layers.sbatch <lengths> [extra args]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-layers
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
echo "lengths=$1 args=${*:2}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
$PY -m bench.layers --lengths "$1" "${@:2}"
uptime
