#!/bin/bash
# KV cache and KDA state check (bench/kvcheck.py): one prompt whole, chunked, chunked on scrambled blocks. Usage:
#   sbatch -p priority --time=00:15:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvcheck.sbatch [L] [chunk]
#SBATCH -J ppm-kvcheck
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppk$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} $(git log --oneline -1)"
uptime
$PY bench/kvcheck.py --prompt-len ${1:-32768} --chunk ${2:-8000}
rc=$?
uptime
rm -rf "$TMPDIR"
exit $rc
