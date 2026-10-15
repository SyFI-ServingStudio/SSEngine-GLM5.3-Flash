#!/bin/bash
# Host-time breakdown of stage 0 (bench/host.py), one GPU. Usage:
#   sbatch -p priority --time=00:25:00 -o measured/logs/%j.out bench/slurm/host.sbatch <lengths> [extra args]
#SBATCH --gres=gpu:1
#SBATCH -J ppm-host
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/pph$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
export PATH=$PATH:/home/kanzhu/.local/bin    # py-spy (appended: ninja stays /usr/bin/ninja)
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
echo "job $SLURM_JOB_ID host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} lengths=$1 args=${*:2}"
uptime
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
$PY bench/host.py --lengths "$1" "${@:2}"
for f in measured/host/${SLURM_JOB_ID}_spy_L*.txt; do [ -f "$f" ] && { echo "== $f"; $PY bench/host.py --spy "$f"; }; done
uptime
rm -rf "$TMPDIR"
