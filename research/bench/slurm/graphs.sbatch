#!/bin/bash
# CUDA graphs, the stages one after another on one GPU (research/bench/graphs.py). Usage, from the repo root:
#   [STAGES="0 1 2 3"] sbatch -p main --time=00:08:00 -o research/measured/logs/%j.out research/bench/slurm/graphs.sbatch
#   [STAGES=...] sbatch -p main --time=00:12:00 -o research/measured/logs/%j.out research/bench/slurm/graphs.sbatch --memory --mb-tokens 32768
# Without arguments: replay vs eager per bucket. With --memory: graph memory beside the full KV pool.
#SBATCH --gres=gpu:1
#SBATCH -J ppm-graphs
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
for s in ${STAGES:-0 1 2 3}; do
  echo "=== stage $s $(date -Is)"
  $PY -m research.bench.graphs --stage $s "$@" || echo "=== stage $s FAILED rc=$?"
done
uptime
