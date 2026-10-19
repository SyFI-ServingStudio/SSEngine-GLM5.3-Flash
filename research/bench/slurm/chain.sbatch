#!/bin/bash
# Model correctness on GPU (research/bench/chain.py). Usage, from the repo root:
#   sbatch -p priority --time=00:30:00 --gres=gpu:4 -o research/measured/logs/%j.out research/bench/slurm/chain.sbatch "chain vllm" [L]
#   sbatch -p priority --time=00:20:00 --gres=gpu:1 -o research/measured/logs/%j.out research/bench/slurm/chain.sbatch split [L]
#SBATCH -J ppm-chain
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
L=${2:-1024}
echo "modes=$1 L=$L"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
for mode in $1; do
  echo "=== $mode $(date -Is)"
  $PY -m research.bench.chain $mode --prompt-len $L || echo "=== $mode FAILED rc=$?"
done
uptime
