#!/bin/bash
# Back-to-back unsynced forwards vs synced (bench/overlap.py), one GPU. Usage:
#   sbatch -p priority --time=00:12:00 -o measured/logs/%j.out bench/slurm/overlap.sbatch [args]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-overlap
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppo$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} args=$*"
uptime
$PY bench/overlap.py "$@"
uptime
rm -rf "$TMPDIR"
