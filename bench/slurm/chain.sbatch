#!/bin/bash
# Model correctness on GPU (bench/chain.py). Usage:
#   sbatch -p priority --time=00:30:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/chain.sbatch "chain vllm" [L]
#   sbatch -p priority --time=00:20:00 --gres=gpu:1 -o measured/logs/%j.out bench/slurm/chain.sbatch split [L]
#SBATCH -J ppm-chain
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppm$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
L=${2:-1024}
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} modes=$1 L=$L"
uptime
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
for mode in $1; do
  echo "=== $mode $(date -Is)"
  $PY bench/chain.py $mode --prompt-len $L || echo "=== $mode FAILED rc=$?"
done
uptime
rm -rf "$TMPDIR"
