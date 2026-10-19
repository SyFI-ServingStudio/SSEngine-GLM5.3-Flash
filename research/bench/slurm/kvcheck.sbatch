#!/bin/bash
# KV cache and KDA state check (research/bench/kvcheck.py): one prompt whole, chunked, chunked on scrambled blocks. Usage, from
# the repo root:
#   sbatch -p priority --time=00:15:00 --gres=gpu:4 -o research/measured/logs/%j.out research/bench/slurm/kvcheck.sbatch [L] [chunk]
#SBATCH -J ppm-kvcheck
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
$PY -m research.bench.kvcheck --prompt-len ${1:-32768} --chunk ${2:-8000}
rc=$?
uptime
exit $rc
