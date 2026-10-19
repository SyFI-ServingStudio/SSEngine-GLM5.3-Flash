#!/bin/bash
# Single-GPU micro-benchmarks, one `python -m bench.<name> <args>` per argument, in order, logs to OUT (default
# research/measured/micro/<job>). Usage, from the repo root:
#   sbatch -p main --gres=gpu:1 -c 28 --mem=200G --time=00:10:00 -o research/measured/logs/%j_micro.out \
#     research/bench/slurm/micro.sbatch "norm_gated_bench" "kpool_expand_bench --rows 4096"
#SBATCH --gres=gpu:1
#SBATCH -J ppm-micro
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
O=$ROOT/${OUT:-measured/micro/$JOB}
mkdir -p "$O"
nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu --format=csv,noheader
rc=0; i=0
for cmd in "$@"; do
  i=$((i + 1)); name=${cmd%% *}
  echo "== $(date -Is) $cmd"
  $PY -m bench.$cmd --out "$O/${i}_${name}.json" > "$O/${i}_${name}.log" 2>&1 || rc=1
  grep -vE "Warning|warn" "$O/${i}_${name}.log" | tail -20
done
uptime
echo "== done rc=$rc"
exit $rc
