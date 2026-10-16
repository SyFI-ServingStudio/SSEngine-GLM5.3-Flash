#!/bin/bash
# Long-context checks (lane longctx), one 4-GPU job each. From the repo root:
#   sbatch -p priority --time=00:15:00 -o measured/runs_lc/slurm/%j.out bench/slurm/longctx.sbatch indexer
#   sbatch -p priority --time=00:15:00 -o measured/runs_lc/slurm/%j.out bench/slurm/longctx.sbatch gpu_tests
# indexer: the topk_canon GPU test, then bench/longctx_indexer.py on GPU 0 (mqa_logits / top-k / topk_canon at 8k-512k context, fork vs our split).
# gpu_tests: the GPU tests of topk_canon and the graph/eager exactness (tests/test_model_gpu.py -k "canonical or graph").
#SBATCH -J ppimpl-lc
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
WHAT=${1:?indexer or gpu_tests}
echo "== $(date -Is) gpus ${CUDA_VISIBLE_DEVICES:-?}: $WHAT"
nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader
case $WHAT in
  indexer) $PY -m pytest -q -m gpu tests/test_model_gpu.py -k canonical; rc=$?
           $PY -m bench.longctx_indexer --out "${OUT:-measured/longctx_indexer.json}"; rc=$(( rc || $? )) ;;
  gpu_tests) $PY -m pytest -q -m gpu tests/test_model_gpu.py -k "${TESTS:-canonical}"; rc=$? ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
