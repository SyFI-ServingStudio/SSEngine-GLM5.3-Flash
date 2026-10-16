#!/bin/bash
#SBATCH -J ppimpl-perf-perm
#SBATCH --gres=gpu:4
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/perf/measured/runs_perf/slurm/%j.out
# Is stage 2's lower steady-state clock (7389: ~1515 MHz vs ~1590 on stages 0/1, all at the 1000 W sw power cap) the
# GPU or the stage? saturated_8k with the stages on the job's GPUs in reverse order (stage s on GPU G[3-s]), with
# nvidia-smi sampling every 100 ms as clock_sat8k.sbatch -> measured/runs_perf/perm/gpu_samples_<job>.csv.
#   sbatch -p priority --time=00:10:00 measured/runs_perf/slurm/perm_sat8k.sbatch
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppg${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
export PPIMPL_REQ_FRONTEND_BIN=/raid/kanzhu/ServingStudio/pp-impl/build/req-frontend/release/session_runner
cd /raid/kanzhu/ServingStudio/pp-impl/lanes/perf
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
R=$PWD/measured/runs_perf/perm
mkdir -p "$R"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
REV=$(echo "$G" | tr ',' '\n' | tac | paste -sd,)
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) gpus $G -> stages on $REV port $P"; uptime
nvidia-smi -i "$G" --query-gpu=index,pci.bus_id,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
nvidia-smi -i "$G" --query-gpu=timestamp,index,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
  --format=csv,nounits -lms 100 > "$R/gpu_samples_${SLURM_JOB_ID}.csv" &
SMI=$!
$PY bench/run.py saturated_8k --inline --client-timeout 900 --gpus "$REV" --port "$P" --out-root "$R" \
  > "$R/${SLURM_JOB_ID}_run.log" 2>&1
rc=$?
sleep 5; kill $SMI
uptime; grep -h " -> " "$R/${SLURM_JOB_ID}_run.log"
echo "== done $(date -Is) rc=$rc"
rm -rf "$TMPDIR"
exit $rc
