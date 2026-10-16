#!/bin/bash
# KV-cache audit (bench/kvaudit/): vLLM TP4+EP vs pp-impl PP4, every layer's cache after prefill. Usage, from the
# repo root (4 GPUs per side; the compare runs on 1 GPU or the CPU):
#   sbatch -p priority --time=00:30:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch vllm 8192
#   sbatch -p priority --time=00:30:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch ours 8192
#   .venv/bin/python -m bench.kvaudit.compare --length 8192 [--device cuda] [--delete]
#   sbatch -p priority --time=00:15:00 --gres=gpu:1 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch probe 131072
# `probe` runs bench/kvaudit/kda_probe.py (KDA layers 0-2, four kernels + float64 on identical inputs) on 1 GPU.
# `ours+compare` runs the compare on GPU 0 after our side when the vLLM dumps are already there.
#SBATCH -J ppm-kvaudit
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
SIDE=$1
L=$2
shift 2
case $SIDE in
  vllm) $PY -m bench.kvaudit.vllm_side --length "$L" "$@"; rc=$? ;;
  ours) $PY -m bench.kvaudit.ours --length "$L" "$@"; rc=$? ;;
  ours+compare)
    $PY -m bench.kvaudit.ours --length "$L" "$@"; rc=$?
    if [ $rc = 0 ]; then $PY -m bench.kvaudit.compare --length "$L" --device cuda; rc=$?; fi ;;
  probe) $PY -m bench.kvaudit.kda_probe --length "$L" "$@"; rc=$? ;;
  fault) $PY -m bench.kvaudit.kda_fault --tokens "$L" "$@"; rc=$? ;;
  compare) $PY -m bench.kvaudit.compare --length "$L" --device cuda "$@"; rc=$? ;;
  *) echo "unknown side $SIDE"; rc=2 ;;
esac
echo "=== kvaudit $SIDE $L rc=$rc $(date -Is)"
uptime
exit $rc
