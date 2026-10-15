#!/bin/bash
# Tune FlashInfer tactics once and verify the file in a fresh process (bench/tune_tactics.py), one GPU. Usage:
#   sbatch -p priority --time=00:15:00 -o measured/logs/%j.out bench/slurm/tactics.sbatch [tune] [verify]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-tactics
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppt$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
for mode in ${@:-tune verify}; do $PY bench/tune_tactics.py $mode; done
uptime
rm -rf "$TMPDIR"
