#!/bin/bash
# M4 (5_prefix_cache.md) runs on one 4-GPU server, into OUT (default measured/runs_m4). Usage, from the repo root:
#   mkdir -p measured/runs_m4/slurm
#   sbatch -p priority -J ppimpl --time=00:12:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch check
#   sbatch -p priority -J ppimpl --time=00:30:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch bringup
#   sbatch -p priority -J ppimpl --time=00:20:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch preflight
#   sbatch -p priority -J ppimpl --time=00:28:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch prefix_n400
# preflight (before any long run), stopping at the first failure: kvcheck at 32,768 tokens (8,192-token chunks must be
# bitwise; 8,000-token chunks leave a 768-token chunk of another mHC split-K count: the batch-numerics baseline), then
# at 40,000 tokens with chunk starts inside 4-token indexer pools and off KDA's 32-token grid (8,001 and 8,190: within
# 2x the baseline per stage, KL within 4x, top-1 equal), prefix_check with P + 1/2/3
# (prefix_check_unaligned), then a smoke of prefix_n400's first ~2 min (client limit 3m incl. ~45 s tokenization:
# prefix_smoke_n400) that must end with failed=0. bench/run.py stops any run at the first fault (server or worker
# traceback, server exit, client failed > 0).
# check: bench/prefix_check.py (round-1 tokens, cache on vs off, ~6 min). bringup: prefix_bringup (20 min) under
# Nsight Systems, traced from microbatch NSYS_MBS (default 564:631: the last round-0 chunk and round 1 of session
# 34, P 512k A 1k, and all of session 35, P 512k A 8k; microbatch 0 is req-frontend's preflight). prefix_n<N>: the
# 20-minute session run at N. BRINGUP_ARGS adds bench/run.py options to bringup, e.g. "--gap-s 17 --sessions 30,31
# --run-name prefix_bringup_rest" (sessions 17 s apart: the cluster stops GPU jobs idle for 10 min). Then `python -m bench.prefix summarize measured/runs_m4 --out measured`.
#SBATCH -J ppimpl
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m4}
mkdir -p "$R/slurm"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
WHAT=${1:?check, preflight, bringup or prefix_n<N>}
echo "== $(date -Is) gpus $G port $P: $WHAT"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
case $WHAT in
  check) $PY -m bench.prefix_check --out "$R" --port "$P"; rc=$? ;;
  preflight)
    $PY -m bench.kvcheck --prompt-len 32768 --chunk 8000 --chunk-b 8192 --exact chunked_b,whole_filler; rc=$?
    (( rc == 0 )) && { $PY -m bench.kvcheck --prompt-len 40000 --chunk 8001 --chunk-b 8190 --noise-len 32768; rc=$?; }
    (( rc == 0 )) && { $PY -m bench.prefix_check --out "$R" --port "$P" --p-shift 1,2,3 --name prefix_check_unaligned; rc=$?; }
    (( rc == 0 )) && { $PY -m bench.run prefix_n400 --inline --gpus "$G" --port "$P" --out-root "$R" --client-limit 3m \
                         --run-name prefix_smoke_n400; rc=$?; }
    ;;
  bringup) $PY -m bench.run prefix_bringup --inline --nsys --nsys-mbs "${NSYS_MBS:-564:631}" --gpus "$G" --port "$P" \
             --out-root "$R" ${BRINGUP_ARGS:-}; rc=$? ;;
  prefix_n*) $PY -m bench.run "$WHAT" --inline --gpus "$G" --port "$P" --out-root "$R"; rc=$? ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
