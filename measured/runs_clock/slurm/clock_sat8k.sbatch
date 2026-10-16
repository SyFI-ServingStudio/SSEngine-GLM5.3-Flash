#!/bin/bash
#SBATCH -J ppimpl-clock
#SBATCH --gres=gpu:4
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/measured/runs_clock/slurm/%j.out
# Do the GPUs clock down under sustained load? saturated_8k on main's defaults, with nvidia-smi sampling SM/memory
# clocks, power, temperature and throttle reasons every 100 ms on the job's GPUs (gpu_samples.csv). Question: stage
# times are ~51 ms for the first microbatches and in one-request runs, ~55-58 ms once the stages run back to back.
#   sbatch -p priority --time=00:10:00 measured/runs_clock/slurm/clock_sat8k.sbatch
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppc${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
cd /raid/kanzhu/ServingStudio/pp-impl
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
R=$PWD/measured/runs_clock
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) gpus $G port $P"; uptime
nvidia-smi -i "$G" --query-gpu=timestamp,index,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 100 > "$R/gpu_samples_${SLURM_JOB_ID}.csv" &
SMI=$!
$PY bench/run.py saturated_8k --inline --client-timeout 900 --gpus "$G" --port "$P" --out-root "$R" \
  > "$R/slurm/${SLURM_JOB_ID}_run.log" 2>&1
rc=$?
sleep 5; kill $SMI
uptime; grep -h " -> " "$R/slurm/${SLURM_JOB_ID}_run.log"
echo "== done $(date -Is) rc=$rc"
rm -rf "$TMPDIR"
exit $rc
