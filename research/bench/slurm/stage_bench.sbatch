#!/bin/bash
# research/bench/stage_bench.py runs on one GPU, one process each, into OUT (default research/measured/stage_bench/<job>). Usage, from
# the repo root:
#   sbatch -p main --gres=gpu:1 -c 28 --mem=200G --time=00:10:00 -o research/measured/logs/%j_stage_bench.out \
#     research/bench/slurm/stage_bench.sbatch "0:time 0:kernels 0:stacks"
# Each item is stage:mode[:spec] (spec: prefix:append,...); STAGE_BENCH_ARGS adds options to every run.
#SBATCH --gres=gpu:1
#SBATCH -J ppm-stagebench
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
O=$ROOT/${OUT:-measured/stage_bench/$JOB}
mkdir -p "$O"
nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu --format=csv,noheader
rc=0; i=0
for item in ${1:-0:time}; do
  s=${item%%:*}; rest=${item#*:}; m=${rest%%:*}; spec=; [[ $rest == *:* ]] && spec=${rest#*:}
  extra=(); [ -n "${spec:-}" ] && extra=(--spec "$spec")
  i=$((i + 1)); tag=${i}_s${s}_${m}
  echo "== $(date -Is) stage $s $m ${spec:-}"
  $PY -m research.bench.stage_bench --stage "$s" --mode "$m" "${extra[@]}" ${STAGE_BENCH_ARGS:-} \
    --out "$O/$tag.json" > "$O/$tag.log" 2>&1 || rc=1
  grep -E "^\[stage_bench\]|Error|error" "$O/$tag.log" | tail -5
done
uptime
echo "== done rc=$rc"
exit $rc
