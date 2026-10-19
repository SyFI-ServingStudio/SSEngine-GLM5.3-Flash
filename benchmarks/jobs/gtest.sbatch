#!/bin/bash
# GPU tests (tests/test_*_gpu.py, marker gpu), selected by a pytest -k expression. Usage, from the repo root:
#   sbatch -p <partition> --time=00:10:00 -o runs/logs/%j.out benchmarks/jobs/gtest.sbatch ["graph_replay or topk_canonical"]
#   sbatch -p <partition> --time=00:20:00 --gres=gpu:4 -o runs/logs/%j.out benchmarks/jobs/gtest.sbatch "transport or serve_http"
# Default: (f) CUDA graph replay vs eager per stage and (g) top-k canonicalization, on one GPU.
#SBATCH --gres=gpu:1
#SBATCH -J ppm-gtest
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/benchmarks/jobs/env.sh"
$PY -m pytest tests -m gpu -k "${1:-graph_replay or topk_canonical}" -v
rc=$?
uptime
exit $rc
