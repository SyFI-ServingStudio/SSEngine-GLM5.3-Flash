#!/bin/bash
#SBATCH -J ppimpl-m3-part
#SBATCH --gres=gpu:4
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/measured/runs_m3/slurm/%j.out
# Part of the final M3 run on the main tree's defaults (transport ce, intake_procs 4, CUDA graphs), one 4-GPU server.
# The run names come from RUNS; partition and --time from the sbatch command line. Same output root as m3_final, so
# `bench/report.py --summarize measured/runs_m3 --out measured` writes measured/iteration.md over all parts.
#   RUNS="one_1k one_2k" sbatch -p priority --time=00:20:00 measured/runs_m3/slurm/m3_part.sbatch
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppr${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
cd ${TREE:-/raid/kanzhu/ServingStudio/pp-impl}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
R=$PWD/measured/runs_m3
JOB=${SLURM_JOB_ID:-dry}
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) host $(hostname) gpus $G port $P runs: $RUNS"
uptime
$PY -c "from ppimpl.config import RunConfig as C; c=C(); print('   defaults:', {k: getattr(c, k, None) for k in ('transport', 'intake_procs', 'cuda_graphs', 'compute_sms', 'max_in_flight')})"
[ -n "${DRY:-}" ] || nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
rc=0; first=1
for n in $RUNS; do
  echo "== $(date -Is) start $n"
  if [ -n "${DRY:-}" ]; then echo "   bench/run.py $n --gpus $G --port $P --out-root $R"; continue; fi
  $PY bench/run.py "$n" --inline --client-timeout 900 --gpus "$G" --port "$P" --out-root "$R" \
    >> "$R/slurm/${JOB}_part.log" 2>&1 || rc=1
  echo "== $(date -Is) done $n rc=$rc"
  uptime
  # the first run failed: the defaults are broken (server start, graphs); stop instead of holding the GPUs
  [ $rc = 1 ] && [ $first = 1 ] && { echo "== first run failed, stopping"; break; }
  first=0
done
grep -h "FAILED\| -> " "$R/slurm/${JOB}_part.log" 2>/dev/null
echo "== done $(date -Is) rc=$rc"
rm -rf "$TMPDIR"
exit $rc
