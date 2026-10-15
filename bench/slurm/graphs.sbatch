#!/bin/bash
# CUDA graph replay vs eager per bucket, the stages one after another on one GPU (bench/graphs.py). Usage:
#   [STAGES="0 1 2 3"] sbatch -p main --time=00:08:00 -o measured/logs/%j.out bench/slurm/graphs.sbatch
#SBATCH --gres=gpu:1
#SBATCH -J ppm-graphs
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppg$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
for s in ${STAGES:-0 1 2 3}; do
  echo "=== stage $s $(date -Is)"
  $PY bench/graphs.py --stage $s || echo "=== stage $s FAILED rc=$?"
done
uptime
rm -rf "$TMPDIR"
