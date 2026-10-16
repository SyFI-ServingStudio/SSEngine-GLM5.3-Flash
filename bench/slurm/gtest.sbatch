#!/bin/bash
# GPU tests (tests/test_*_gpu.py, marker gpu), selected by a pytest -k expression. Usage, from the repo root:
#   sbatch -p main --time=00:10:00 -o measured/logs/%j.out bench/slurm/gtest.sbatch ["graph_replay or topk_canonical"]
#   sbatch -p main --time=00:20:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/gtest.sbatch "transport or serve_http"
# Default: (f) CUDA graph replay vs eager per stage and (g) top-k canonicalization, on one GPU.
#SBATCH --gres=gpu:1
#SBATCH -J ppm-gtest
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
$PY -m pytest tests -m gpu -k "${1:-graph_replay or topk_canonical}" -v
rc=$?
uptime
exit $rc
