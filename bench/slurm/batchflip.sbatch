#!/bin/bash
# One request under different batch mates and block histories, the 4 stages one after another on one GPU
# (bench/batchflip.py). Usage:
#   [STAGES="0"] sbatch -p priority --time=00:15:00 -o measured/logs/%j.out bench/slurm/batchflip.sbatch [--moe-probe]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-batchflip
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppb$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?}"
uptime
for s in ${STAGES:-0 1 2 3}; do
  $PY bench/batchflip.py --stage $s --prompts measured/batchflip/prompts.npz --work "$TMPDIR/work" \
    --out measured/batchflip/${SLURM_JOB_ID}_s$s.json "$@"
done
uptime
rm -rf "$TMPDIR"
