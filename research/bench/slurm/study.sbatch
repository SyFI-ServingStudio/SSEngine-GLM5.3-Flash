#!/bin/bash
# One-off studies: benchmarks/run.py runs on one 4-GPU server, each with its own tag, arguments and environment, optionally
# with GPU clock/power sampling or the stages on the GPUs in reverse order. Usage, from the repo root:
#   STUDY="tag:[VAR=value ...] <run.py args>;..." [SAMPLE=1] [REVERSE=1] [OUT=research/measured/runs_study] \
#     sbatch -p priority --time=00:20:00 -o <OUT>/slurm/%j.out research/bench/slurm/study.sbatch
# Each run goes to <OUT>/<tag>/ and its log to <OUT>/slurm/<job>_<tag>.log. SAMPLE=1: nvidia-smi samples SM/memory
# clocks, power, temperature and throttle reasons every 100 ms into <OUT>/gpu_samples_<job>.csv (research/bench/clock_align.py
# aligns them with microbatches). REVERSE=1: stage s on GPU G[3-s]. Studies run this way (research/measured/issues.md):
#   OUT=research/measured/runs_clock SAMPLE=1 STUDY="sat8k:saturated_8k"                                  # power cap (7389)
#   OUT=research/measured/runs_intake STUDY="n8:saturated_8k --intake-procs 8;n16:saturated_8k --intake-procs 16;\
#     n16_1k:saturated_1k --intake-procs 16"                                                    # first TTFT (7390)
#   OUT=research/measured/runs_perf STUDY="early:saturated_8k --nsys --nsys-mbs 0:10;\
#     steady:saturated_8k --nsys --nsys-mbs 1000:1010;one:one_8k --nsys;\
#     sync:PPIMPL_SYNC_EACH_MB=1 saturated_8k"                                                  # kernel classes (7392)
#   OUT=research/measured/runs_perf SAMPLE=1 REVERSE=1 STUDY="perm:saturated_8k"                          # GPU placement (7396)
#SBATCH -J ppimpl-study
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
R=$ROOT/${OUT:-measured/runs_study}
mkdir -p "$R/slurm"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
[ -n "${REVERSE:-}" ] && G=$(echo "$G" | tr ',' '\n' | tac | paste -sd,)
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) gpus $G port $P study: $STUDY"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
SMI=
if [ -n "${SAMPLE:-}" ]; then
  nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --query-gpu=timestamp,index,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
    --format=csv,nounits -lms 100 > "$R/gpu_samples_$JOB.csv" &
  SMI=$!
fi
rc=0
off=0
IFS=';' read -ra ITEMS <<< "$STUDY"
for it in "${ITEMS[@]}"; do
  tag=${it%%:*}
  read -ra words <<< "${it#*:}"
  envs=()
  while [ ${#words[@]} -gt 0 ] && [[ ${words[0]} == [A-Z]*=* ]]; do envs+=("${words[0]}"); words=("${words[@]:1}"); done
  echo "== $(date -Is) start $tag: ${envs[*]} ${words[*]}"; uptime
  env "${envs[@]}" $PY benchmarks/run.py "${words[@]}" --inline --client-timeout 900 --gpus "$G" --port $((P + off)) \
    --out-root "$R/$tag" >> "$R/slurm/${JOB}_$tag.log" 2>&1 || rc=1
  echo "== $(date -Is) done $tag rc=$rc"; uptime
  off=$((off + 20))
done
[ -n "$SMI" ] && { sleep 5; kill "$SMI"; }
grep -h "FAILED\| -> " "$R"/slurm/"${JOB}"_*.log 2>/dev/null
echo "== done $(date -Is) rc=$rc"
exit $rc
