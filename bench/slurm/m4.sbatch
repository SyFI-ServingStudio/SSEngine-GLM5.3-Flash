#!/bin/bash
# M4 (5_prefix_cache.md) runs on one 4-GPU server, into OUT (default measured/runs_m4). Usage, from the repo root:
#   mkdir -p measured/runs_m4/slurm
#   sbatch -p priority --time=00:12:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch check
#   sbatch -p priority --time=00:30:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch bringup
#   sbatch -p priority --time=00:28:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch prefix_n400
# check: bench/prefix_check.py (round-1 tokens, cache on vs off, ~6 min). bringup: prefix_bringup (20 min) under
# Nsight Systems, traced from microbatch NSYS_MBS (default 564:631: the last round-0 chunk and round 1 of session
# 34, P 512k A 1k, and all of session 35, P 512k A 8k; microbatch 0 is req-frontend's preflight). prefix_n<N>: the
# 20-minute session run at N. Then `python -m bench.prefix summarize measured/runs_m4 --out measured`.
#SBATCH -J ppm4
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m4}
mkdir -p "$R/slurm"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
WHAT=${1:?check, bringup or prefix_n<N>}
echo "== $(date -Is) gpus $G port $P: $WHAT"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
case $WHAT in
  check) $PY -m bench.prefix_check --out "$R" --port "$P"; rc=$? ;;
  bringup) $PY -m bench.run prefix_bringup --inline --nsys --nsys-mbs "${NSYS_MBS:-564:631}" --gpus "$G" --port "$P" \
             --out-root "$R"; rc=$? ;;
  prefix_n*) $PY -m bench.run "$WHAT" --inline --gpus "$G" --port "$P" --out-root "$R"; rc=$? ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
