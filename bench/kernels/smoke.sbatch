#!/bin/bash
#SBATCH -p priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -J pp-remeasure-smoke
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/measure/measured/kernels_runs/logs/%x-%j.out
# One smallest spec of each (kind, backend) in the re-measure queue, through the same wrapper,
# so an env or import failure shows in minutes. Exits nonzero if any call returned no ok result.
set -u
LANE=/raid/kanzhu/ServingStudio/pp-impl/lanes/measure
SIM=/raid/kanzhu/ServingStudio/wt-pp-main
Q=$LANE/measured/kernels_runs/queue
OUT=$LANE/measured/kernels_runs/smoke/$SLURM_JOB_ID
mkdir -p "$OUT"
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export MOESIM_CUPTI_BUILD_DIR=/raid/kanzhu/ServingStudio/tmp/moesim_cupti_ext_ppimpl
export VIBESIM_AUTOTUNE_CACHE_DIR=$Q/autotune
echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) uptime: $(uptime)"
bad=0
for f in $(ls "$Q/todo" | grep -v rows | awk -F'__' '!seen[$2"__"$3]++'); do
  stem=${f%.jsonl}; kind=$(echo "$stem" | awk -F'__' '{print $2}'); backend=$(echo "$stem" | awk -F'__' '{print $3}')
  python3 "$LANE/bench/kernels/smallest_spec.py" "$Q/todo/$stem.rows.jsonl" > "$OUT/$kind.$backend.spec.jsonl"
  t0=$(date +%s)
  ( cd "$SIM" && PPM_KNAMES_DIR="$OUT/knames_$kind.$backend" timeout 240 uv run --no-sync python "$LANE/bench/kernels/ppimpl_profile.py" run "$kind" \
      --backend "$backend" --specs "$OUT/$kind.$backend.spec.jsonl" --fresh --gpu-name "NVIDIA B200" --json \
      > "$OUT/$kind.$backend.json" 2> "$OUT/$kind.$backend.err" )
  rc=$?
  st=$(python3 -c "import json; d=json.load(open('$OUT/$kind.$backend.json')); r=d['results'][0]; print(r.get('status'), r.get('metrics',{}).get('time_ms'), (r.get('error') or '')[:200])" 2>/dev/null || echo "no-json")
  echo "$kind/$backend rc=$rc $(( $(date +%s) - t0 ))s: $st"
  case "$st" in ok*) ;; *) bad=1; tail -15 "$OUT/$kind.$backend.err";; esac
done
echo "end $(date -Is) bad=$bad"
exit $bad
