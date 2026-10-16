#!/bin/bash
# Long-context checks (lane longctx), one 4-GPU job each. From the repo root:
#   sbatch -p priority --time=00:15:00 -o measured/runs_lc/slurm/%j.out bench/slurm/longctx.sbatch indexer
#   sbatch -p priority --time=00:15:00 -o measured/runs_lc/slurm/%j.out bench/slurm/longctx.sbatch gpu_tests
# indexer: the topk_canon GPU test, then bench/longctx_indexer.py on GPU 0 (mqa_logits / top-k / topk_canon at 8k-512k context, fork vs our split).
# gpu_tests: the GPU tests of topk_canon and the graph/eager exactness (tests/test_model_gpu.py -k "canonical or graph").
# topk_det: bench/topk_determinism.py (is DeepSelect's top-k already canonical?), GPU 0.
# runs: the bench.run commands in RUNS (";"-separated argument lists, e.g. "prefix_bringup --trace X --run-name a;
#   prefix_bringup --trace Y --run-name b"), each on a fresh server, one after another, into OUT (default
#   measured/runs_lc); stops at the first failure. Uptime before and after each.
#SBATCH -J ppimpl-lc
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
WHAT=${1:?indexer, topk_det, gpu_tests or runs}
echo "== $(date -Is) gpus ${CUDA_VISIBLE_DEVICES:-?}: $WHAT"
nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader
case $WHAT in
  indexer) $PY -m pytest -q -m gpu tests/test_model_gpu.py -k canonical; rc=$?
           $PY -m bench.longctx_indexer --out "${OUT:-measured/longctx_indexer.json}"; rc=$(( rc || $? )) ;;
  topk_det) $PY -m bench.topk_determinism --out "${OUT:-measured/topk_determinism.json}"; rc=$? ;;
  gpu_tests) $PY -m pytest -q -m gpu tests/test_model_gpu.py -k "${TESTS:-canonical}"; rc=$? ;;
  runs) G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}; P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100)); rc=0
        IFS=';' read -ra CMDS <<< "${RUNS:?RUNS}"
        for c in "${CMDS[@]}"; do
          echo "== $(date -Is) uptime $(uptime): $c"
          $PY -m bench.run $c --inline --gpus "$G" --port "$P" --out-root "${OUT:-measured/runs_lc}"; rc=$?
          echo "== $(date -Is) rc=$rc uptime $(uptime)"
          (( rc == 0 )) || break
        done ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
