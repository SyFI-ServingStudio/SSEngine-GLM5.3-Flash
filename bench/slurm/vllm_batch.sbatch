#!/bin/bash
# vLLM TP4+EP: uni_1k_3 alone and in 8 x 1k batches (bench/vllm_batch.py). Usage:
#   sbatch -p priority --time=00:20:00 -o measured/logs/%j.out bench/slurm/vllm_batch.sbatch
#SBATCH --gres=gpu:4
#SBATCH -J ppm-vllmbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppv$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
$PY bench/vllm_batch.py --prompts measured/batchflip/prompts.npz --out measured/batchflip/vllm_${SLURM_JOB_ID}.json
uptime
rm -rf "$TMPDIR"
