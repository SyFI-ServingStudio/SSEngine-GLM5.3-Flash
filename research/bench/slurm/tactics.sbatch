#!/bin/bash
# Tune FlashInfer tactics once and verify the file in a fresh process (benchmarks/tune_tactics.py), one GPU. Usage, from
# the repo root:
#   sbatch -p priority --time=00:15:00 -o research/measured/logs/%j.out research/bench/slurm/tactics.sbatch [tune] [verify]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-tactics
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
for mode in ${@:-tune verify}; do $PY -m benchmarks.tune_tactics $mode; done
uptime
