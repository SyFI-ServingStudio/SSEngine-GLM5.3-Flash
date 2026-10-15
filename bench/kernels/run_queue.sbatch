#!/bin/bash
#SBATCH -p main
#SBATCH --gres=gpu:2
#SBATCH --time=00:27:00
#SBATCH -J pp-remeasure
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/measure/measured/kernels_runs/logs/%x-%j.out
# Re-measure kernel rows (2_kernels.md §4/§8) through the ServingStudioSim profiler runners
# executing in pp-impl/.venv (bench/kernels/ppimpl_profile.py), cold L2, --fresh (no DB writes).
# One lane per GPU claims chunks from the shared queue until the job's time runs short;
# several such jobs drain one queue. LIMIT_S must match --time.
set -u
LIMIT_S=${LIMIT_S:-1620}
LANE=/raid/kanzhu/ServingStudio/pp-impl/lanes/measure
SIM=/raid/kanzhu/ServingStudio/wt-pp-main
Q=$LANE/measured/kernels_runs/queue
LOG=$LANE/measured/kernels_runs/logs/$SLURM_JOB_ID
mkdir -p "$LOG" "$Q/out" "$Q/knames"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export MOESIM_CUPTI_BUILD_DIR=/raid/kanzhu/ServingStudio/tmp/moesim_cupti_ext_ppimpl
export VIBESIM_AUTOTUNE_CACHE_DIR=$Q/autotune
mkdir -p "$VIBESIM_AUTOTUNE_CACHE_DIR"
START=$(date +%s); DEADLINE=$((START + LIMIT_S))
{
  echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) limit_s=$LIMIT_S"
  echo "uptime: $(uptime)"
  echo "sim_head=$(git -C $SIM rev-parse HEAD) lane_head=$(git -C $LANE rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
  ls $MOESIM_CUPTI_BUILD_DIR/*.so
} > "$LOG/env.txt" 2>&1

lane() {
  local dev=$1
  while :; do
    local left=$(( DEADLINE - $(date +%s) ))
    (( left < 480 )) && { echo "$(date -Is) dev$dev stop: ${left}s left" >> "$LOG/progress.log"; return; }
    local f; f=$(ls "$Q/todo" 2>/dev/null | grep -v '\.rows\.jsonl$' | sort | head -1)
    [ -z "$f" ] && { echo "$(date -Is) dev$dev queue empty" >> "$LOG/progress.log"; return; }
    local stem=${f%.jsonl}
    mv "$Q/todo/$f" "$Q/claimed/$f" 2>/dev/null || continue   # another lane took it
    mv "$Q/todo/$stem.rows.jsonl" "$Q/claimed/" 2>/dev/null
    local kind backend; kind=$(echo "$stem" | awk -F'__' '{print $2}'); backend=$(echo "$stem" | awk -F'__' '{print $3}')
    local n; n=$(wc -l < "$Q/claimed/$f")
    echo "$(date -Is) dev$dev start $stem n=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$LOG/progress.log"
    mkdir -p "$Q/knames/$stem"
    ( cd "$SIM" && CUDA_VISIBLE_DEVICES=$dev PPM_KNAMES_DIR="$Q/knames/$stem" \
        timeout $(( left - 120 )) uv run --no-sync python "$LANE/bench/kernels/ppimpl_profile.py" run "$kind" \
        --backend "$backend" --specs "$Q/claimed/$f" --fresh --gpu-name "NVIDIA B200" --json \
        > "$Q/out/$stem.json" 2> "$Q/out/$stem.err" )
    local rc=$?
    if [ $rc -eq 0 ] || grep -q '"results"' "$Q/out/$stem.json"; then
      mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/done/"
    elif [ $rc -eq 124 ]; then
      mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/todo/"   # out of time: back to the queue
    else
      mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/failed/"
    fi
    echo "$(date -Is) dev$dev end $stem rc=$rc" >> "$LOG/progress.log"
  done
}
IFS=',' read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
for d in "${DEVS[@]}"; do lane "$d" & done
wait
echo "$(date -Is) job done; uptime: $(uptime)" >> "$LOG/progress.log"
