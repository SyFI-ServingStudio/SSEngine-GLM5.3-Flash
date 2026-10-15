#!/bin/bash
#SBATCH -p priority
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH -J pp-remeasure
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/measure/measured/kernels_runs/logs/%x-%j.out
# Re-measure kernel rows (2_kernels.md §4/§8) through the ServingStudioSim profiler runners
# executing in pp-impl/.venv (bench/kernels/ppimpl_profile.py), cold L2, --fresh (no DB writes).
# One lane per GPU claims chunks from the shared queue until the job's time runs short;
# several such jobs drain one queue. The deadline comes from the job's own time limit.
set -u
TL=$(squeue -h -j "$SLURM_JOB_ID" -o %l)   # [[D-]HH:]MM:SS
LIMIT_S=$(echo "$TL" | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2)s+=3600*$(n-2); if(n>3)s+=86400*$(n-3); print s}')
LANE=/raid/kanzhu/ServingStudio/pp-impl/lanes/measure
SIM=/raid/kanzhu/ServingStudio/wt-pp-main
Q=$LANE/measured/kernels_runs/queue
LOG=$LANE/measured/kernels_runs/logs/$SLURM_JOB_ID
mkdir -p "$LOG" "$Q/out" "$Q/knames" "$Q/timeouts"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export MOESIM_CUPTI_BUILD_DIR=/raid/kanzhu/ServingStudio/tmp/moesim_cupti_ext_ppimpl
export VIBESIM_AUTOTUNE_CACHE_DIR=$Q/autotune
mkdir -p "$VIBESIM_AUTOTUNE_CACHE_DIR"
START=$(date +%s); DEADLINE=$((START + LIMIT_S))
NEED_S=${NEED_S:-240}; NEED_MOE_S=${NEED_MOE_S:-360}
{
  echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) limit_s=$LIMIT_S"
  echo "uptime: $(uptime)"
  echo "sim_head=$(git -C $SIM rev-parse HEAD) lane_head=$(git -C $LANE rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
  ls $MOESIM_CUPTI_BUILD_DIR/*.so
} > "$LOG/env.txt" 2>&1

lane() {
  local dev=$1 fails=0
  while :; do
    (( fails >= 2 )) && { echo "$(date -Is) dev$dev stop: 2 chunks in a row returned no results" >> "$LOG/progress.log"; return; }
    local left=$(( DEADLINE - $(date +%s) ))
    # A chunk needs NEED_S seconds of job left to start (MoE chunks autotune first: NEED_MOE_S);
    # take the first chunk in queue order that fits, else stop.
    (( left < NEED_S )) && { echo "$(date -Is) dev$dev stop: ${left}s left" >> "$LOG/progress.log"; return; }
    local f="" c
    for c in $(ls "$Q/todo" 2>/dev/null | grep -v '\.rows\.jsonl$' | sort); do
      case "$c" in *__nvfp4_fused_moe__*) (( left < NEED_MOE_S )) && continue;; esac
      f=$c; break
    done
    if [ -z "$f" ]; then
      if ls "$Q/todo" 2>/dev/null | grep -q .; then
        echo "$(date -Is) dev$dev stop: ${left}s left, too little for the remaining chunks" >> "$LOG/progress.log"
      else
        echo "$(date -Is) dev$dev queue empty" >> "$LOG/progress.log"
      fi
      return
    fi
    local stem=${f%.jsonl}
    mv "$Q/todo/$f" "$Q/claimed/$f" 2>/dev/null || continue   # another lane took it
    mv "$Q/todo/$stem.rows.jsonl" "$Q/claimed/" 2>/dev/null
    local kind backend; kind=$(echo "$stem" | awk -F'__' '{print $2}'); backend=$(echo "$stem" | awk -F'__' '{print $3}')
    local n; n=$(wc -l < "$Q/claimed/$f")
    echo "$(date -Is) dev$dev start $stem n=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$LOG/progress.log"
    mkdir -p "$Q/knames/$stem"
    # run_call.sh: own session; on timeout it kills that session only (never by GPU or user).
    ( cd "$SIM" && CUDA_VISIBLE_DEVICES=$dev PPM_KNAMES_DIR="$Q/knames/$stem" \
        "$LANE/bench/kernels/run_call.sh" $(( left - 100 )) "$Q/out/$stem.json" "$Q/out/$stem.err" \
        uv run --no-sync python "$LANE/bench/kernels/ppimpl_profile.py" run "$kind" \
        --backend "$backend" --specs "$Q/claimed/$f" --fresh --gpu-name "NVIDIA B200" --json )
    local rc=$?
    if [ $rc -eq 0 ] || grep -q '"results"' "$Q/out/$stem.json"; then
      fails=0
      mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/done/"
    elif [ $rc -eq 124 ] || [ $rc -eq 125 ]; then
      # Out of time (or the GPU was not released): back to the queue once, then to failed.
      if [ -e "$Q/timeouts/$stem" ]; then
        mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/failed/"
      else
        touch "$Q/timeouts/$stem"; mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/todo/"
      fi
      { echo "--- $stem rc=$rc (timeout/GPU not released) stderr tail:"; tail -5 "$Q/out/$stem.err"; } >> "$LOG/errors.log"
      if [ $rc -eq 125 ]; then
        echo "$(date -Is) dev$dev stop: GPU still held after cleanup" >> "$LOG/progress.log"; return
      fi
    else
      fails=$((fails + 1))
      mv "$Q/claimed/$f" "$Q/claimed/$stem.rows.jsonl" "$Q/failed/"
      { echo "--- $stem rc=$rc stderr tail ($Q/out/$stem.err):"; tail -20 "$Q/out/$stem.err"; } >> "$LOG/errors.log"
    fi
    echo "$(date -Is) dev$dev end $stem rc=$rc" >> "$LOG/progress.log"
  done
}
IFS=',' read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
for d in "${DEVS[@]}"; do lane "$d" & done
wait
echo "$(date -Is) job done; uptime: $(uptime)" >> "$LOG/progress.log"
