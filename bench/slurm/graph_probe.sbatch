#!/bin/bash
# Which CUDA graph bucket breaks capture, and does the warm-up pass on its own (lane longctx). From the repo root:
#   sbatch -p priority --time=00:25:00 -o measured/runs_lc/slurm/%j.out bench/slurm/graph_probe.sbatch
# Waves of 4 processes, one GPU each (bench/graph_probe.py), then bench/graphs.py on every stage (the GPU test
# test_graph_replay_matches_eager, which builds each stage with warm-up and captures GRAPH_BUCKETS).
#SBATCH -J ppimpl-lc-probe
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUT=measured/runs_lc/probe_$JOB; mkdir -p $OUT
echo "== $(date -Is) gpus ${G[*]} $(git log --oneline -1 2>/dev/null) uptime $(uptime)"
wave() {   # wave name "args0" "args1" ...: one process per GPU, wait for all
  local name=$1; shift; local i=0 pids=()
  for args in "$@"; do
    echo "== $(date -Is) $name gpu ${G[$i]}: $args"
    env CUDA_VISIBLE_DEVICES=${G[$i]} $args > $OUT/${name}_$i.log 2>&1 & pids+=($!); i=$((i + 1))
  done
  for j in "${!pids[@]}"; do wait ${pids[$j]}; echo "   $name $j rc=$? $(grep -h '\[graph_probe\]\|\[graphs\]\|Error:\|error:' $OUT/${name}_$j.log | tail -2 | cut -c1-300)"; done
}
wave w1 "$PY -m bench.graph_probe --stage 0 --buckets 7168" "$PY -m bench.graph_probe --stage 0 --buckets 6144" \
        "$PY -m bench.graph_probe --stage 0 --buckets 5120" "$PY -m bench.graph_probe --stage 0 --buckets 3072"
wave w2 "$PY -m bench.graph_probe --stage 0 --buckets 1536" "$PY -m bench.graph_probe --stage 0 --buckets 8192" \
        "env CUDA_LAUNCH_BLOCKING=1 $PY -m bench.graph_probe --stage 0 --eager 7168" \
        "env CUDA_LAUNCH_BLOCKING=1 $PY -m bench.graph_probe --stage 0 --eager 3000"
wave graphs "$PY -m bench.graphs --stage 0 --out $OUT/graphs_s0.json" "$PY -m bench.graphs --stage 1 --out $OUT/graphs_s1.json" \
        "$PY -m bench.graphs --stage 2 --out $OUT/graphs_s2.json" "$PY -m bench.graphs --stage 3 --out $OUT/graphs_s3.json"
for s in 0 1 2 3; do python3 -c "import json,sys; r=json.load(open('$OUT/graphs_s$s.json')); print('graphs stage $s pass', r['pass'])" 2>&1 | tail -1; done
echo "== done $(date -Is) uptime $(uptime)"
