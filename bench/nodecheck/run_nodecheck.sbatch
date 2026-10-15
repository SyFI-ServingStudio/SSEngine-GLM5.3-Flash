#!/bin/bash
#SBATCH -p priority
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -J pp-nodecheck
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/measure/measured/nodecheck_runs/%x-%j.out
# Node check (2_kernels.md §4): re-measure the stratified sample with the
# ServingStudioSim profiler (wt-pp-main) in --fresh mode against a scratch copy
# of profile.db. One lane per allocated GPU takes the (kind, backend) calls.
# NC_ONLY: optional regex over "<kind>\t<backend>" to re-run a subset of calls;
# NC_SKIP_H2D=1 skips the H2D bandwidth measurement.
set -u
LANE=/raid/kanzhu/ServingStudio/pp-impl/lanes/measure
SIM=/raid/kanzhu/ServingStudio/wt-pp-main
SPECS=$LANE/measured/nodecheck_specs
OUT=$LANE/measured/nodecheck_runs/$SLURM_JOB_ID
mkdir -p "$OUT"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export VIBESIM_VLLM_FORK_ROOT=/raid/kanzhu/ServingStudio/wt-flash-dflash2/alignment/profiler/vllm
cp "$SIM/profiling/profile.db" "$OUT/scratch_profile.db"
cd "$SIM"
{
  echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) host=$(hostname)"
  echo "sim_head=$(git rev-parse HEAD) dirty_files=$(git status --porcelain --untracked-files=no | wc -l) fork=$(git -C $VIBESIM_VLLM_FORK_ROOT rev-parse HEAD)"
  echo "uptime: $(uptime)"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,driver_version --format=csv
  nvidia-smi topo -m
} > "$OUT/env.txt" 2>&1
# Host DRAM -> HBM bandwidth for the setup record (pinned memory, fork venv torch).
[ "${NC_SKIP_H2D:-0}" = 1 ] || timeout 300 $VIBESIM_VLLM_FORK_ROOT/.venv/bin/python $LANE/bench/nodecheck/h2d_bw.py > "$OUT/h2d_bw.txt" 2>&1

IFS=',' read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
NL=${#DEVS[@]}
# Balance calls over the lanes by spec count (largest first).
grep -E "${NC_ONLY:-.}" "$SPECS/calls.tsv" | sort -t$'\t' -k4,4nr > "$OUT/calls_sorted.tsv"
declare -a LOAD=()
for (( i = 0; i < NL; i++ )); do LOAD[i]=0; : > "$OUT/lane$i.tsv"; done
while IFS=$'\t' read -r kind backend file n; do
  best=0; for (( i = 1; i < NL; i++ )); do (( LOAD[i] < LOAD[best] )) && best=$i; done
  # Weight each call by its CLI startup as well as its specs.
  LOAD[best]=$(( LOAD[best] + n + 4 ))
  printf '%s\t%s\t%s\t%s\n' "$kind" "$backend" "$file" "$n" >> "$OUT/lane$best.tsv"
done < "$OUT/calls_sorted.tsv"

run_lane() {
  local i=$1 dev=$2
  while IFS=$'\t' read -r kind backend file n; do
    local tag="${kind}__${backend}"
    echo "$(date -Is) lane$i dev=$dev start $tag n=$n uptime=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"
    # run_call.sh: own session; on timeout it kills that session only.
    CUDA_VISIBLE_DEVICES=$dev "$LANE/bench/kernels/run_call.sh" 1200 "$OUT/$tag.json" "$OUT/$tag.err" \
      uv run --no-sync python -m launcher kernel-profile run "$kind" \
      --backend "$backend" --specs "$SPECS/$file" --fresh --gpu-name "NVIDIA B200" \
      --db "$OUT/scratch_profile.db" --json
    echo "$(date -Is) lane$i dev=$dev done $tag rc=$?" >> "$OUT/progress.log"
  done < "$OUT/lane$i.tsv"
}
for (( i = 0; i < NL; i++ )); do run_lane $i "${DEVS[$i]}" & done
wait
echo "$(date -Is) all lanes done; uptime: $(uptime)" >> "$OUT/progress.log"
rm -f "$OUT/scratch_profile.db"
