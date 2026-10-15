#!/bin/bash
# CUDA graphs: memory beside the full KV pool (bench/graphs.py --memory), the stages one after another on one GPU.
# Usage: [STAGES="0 1 2 3"] [MB=32768] sbatch -p main --time=00:12:00 -o measured/logs/%j.out bench/slurm/graphs_mem.sbatch
#SBATCH --gres=gpu:1
#SBATCH -J ppm-gmem
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppgm$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
for s in ${STAGES:-0 1 2 3}; do
  echo "=== stage $s $(date -Is)"
  $PY bench/graphs.py --stage $s --memory --mb-tokens ${MB:-32768} || echo "=== stage $s FAILED rc=$?"
done
rm -rf "$TMPDIR"
