#!/bin/bash
#SBATCH -J ppimpl-intake
#SBATCH --gres=gpu:4
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/measured/runs_intake/slurm/%j.out
# First-request TTFT vs intake processes. RUNS = "name:intake_procs ...", each into measured/runs_intake/n<N>/.
# 7368 saturated_8k (intake 4): token ready at 209 ms, response written at 415 ms (the intake's loop parses other
# requests' bodies meanwhile). saturated_1k: response written at 226 ms, client read it at 1401 ms (client-bound).
#   RUNS="saturated_8k:8 saturated_8k:16 saturated_1k:16" sbatch -p priority --time=00:20:00 .../intake_sweep.sbatch
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppi${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
cd /raid/kanzhu/ServingStudio/pp-impl
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
R=$PWD/measured/runs_intake
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) gpus $G port $P runs: $RUNS"; uptime
rc=0
for x in $RUNS; do
  n=${x%%:*}; k=${x##*:}
  echo "== $(date -Is) start $n intake_procs $k"
  $PY bench/run.py "$n" --inline --client-timeout 900 --gpus "$G" --port "$P" --intake-procs "$k" \
    --out-root "$R/n$k" >> "$R/slurm/${SLURM_JOB_ID}.log" 2>&1 || rc=1
  echo "== $(date -Is) done $n intake_procs $k rc=$rc"; uptime
done
echo "== done rc=$rc"; rm -rf "$TMPDIR"; exit $rc
