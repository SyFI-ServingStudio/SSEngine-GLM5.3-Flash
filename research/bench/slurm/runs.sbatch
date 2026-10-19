#!/bin/bash
# Handoff runs (benchmarks/run.py) on one 4-GPU server, one run name after another, on the repo's defaults (CE transport,
# intake_procs 16, CUDA graphs). Run names come from RUNS, partition and --time from the sbatch command line; the runs
# go to OUT (default research/measured/runs_m3), so `python -m benchmarks.report --summarize research/measured/runs_m3 --out measured` writes
# research/measured/iteration.md over every job's runs. Usage, from the repo root:
#   mkdir -p research/measured/runs_m3/slurm
#   RUNS="one_1k one_2k" sbatch -p priority --time=00:20:00 -o research/measured/runs_m3/slurm/%j.out research/bench/slurm/runs.sbatch
# DRY=1 bash research/bench/slurm/runs.sbatch prints the plan only. HTTP ports come from the job id; the first run failing
# stops the job (the defaults are broken: server start, graphs) instead of holding the GPUs.
#SBATCH -J ppimpl-runs
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
R=$ROOT/${OUT:-measured/runs_m3}
mkdir -p "$R/slurm"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) gpus $G port $P runs: $RUNS"
$PY -c "from ppimpl.config import RunConfig as C; c=C(); print('   defaults:', {k: getattr(c, k) for k in ('intake_procs', 'cuda_graphs', 'max_in_flight')})"
[ -n "${DRY:-}" ] || nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
rc=0; first=1
for n in $RUNS; do
  echo "== $(date -Is) start $n"
  if [ -n "${DRY:-}" ]; then echo "   python -m benchmarks.run $n --gpus $G --port $P --out-root $R"; continue; fi
  $PY -m benchmarks.run "$n" --inline --client-timeout 900 --gpus "$G" --port "$P" --out-root "$R" \
    >> "$R/slurm/${JOB}_part.log" 2>&1 || rc=1
  echo "== $(date -Is) done $n rc=$rc"
  uptime
  [ $rc = 1 ] && [ $first = 1 ] && { echo "== first run failed, stopping"; break; }
  first=0
done
grep -h "FAILED\| -> " "$R/slurm/${JOB}_part.log" 2>/dev/null
echo "== done $(date -Is) rc=$rc"
exit $rc
