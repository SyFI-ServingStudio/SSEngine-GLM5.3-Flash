#!/bin/bash
# GPU tests (f) CUDA graph replay vs eager per stage and (g) top-k canonicalization (tests/test_model_gpu.py). Usage:
#   sbatch -p main --time=00:10:00 -o measured/logs/%j.out bench/slurm/gtest.sbatch
#SBATCH --gres=gpu:1
#SBATCH -J ppm-gtest
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppt$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
$PY -m pytest tests/test_model_gpu.py -m gpu -k "graph_replay or topk_canonical" -v
rc=$?
uptime
rm -rf "$TMPDIR"
exit $rc
