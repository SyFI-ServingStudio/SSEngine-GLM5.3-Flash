#!/bin/bash
# M1 correctness (2_kernels.md §5). Usage, from the repo root:
#   sbatch -p main --time=00:25:00 -o research/measured/logs/%j.out research/bench/slurm/kernels_correctness.sbatch [rows] [configs]
# rows: smallest,largest (default); configs: comma-separated numbers (default: all 35).
#SBATCH --gres=gpu:1
#SBATCH -J ppm-kcorr
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
echo "rows=${1:-smallest,largest} configs=${2:-}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv
$PY -m research.bench.kernels_correctness --out research/measured/kernels_correctness.jsonl --rows "${1:-smallest,largest}" --configs "${2:-}"
uptime
