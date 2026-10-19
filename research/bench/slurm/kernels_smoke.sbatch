#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -J pp-remeasure-smoke
# One smallest spec of each (kind, backend) in the re-measure queue, through the same wrapper,
# so an env or import failure shows in minutes. Exits nonzero if any call returned no ok result. Usage, from the
# repo root:
#   sbatch -p main -o research/measured/logs/%x-%j.out research/bench/slurm/kernels_smoke.sbatch
# PPIMPL_SIM: a ServingStudioSim checkout of pp-main with its profiler venv (`uv sync` there); default
# <main tree>/third_party/ServingStudioSim (benchmarks/jobs/env.sh).
set -u
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
SIM=${PPIMPL_SIM:-$MAIN/third_party/ServingStudioSim}
Q=$ROOT/measured/kernels_runs/queue
OUT=$ROOT/measured/kernels_runs/smoke/$SLURM_JOB_ID
mkdir -p "$OUT"
export MOESIM_CUPTI_BUILD_DIR=${MOESIM_CUPTI_BUILD_DIR:-$MAIN/build/moesim_cupti_ext}
export VIBESIM_AUTOTUNE_CACHE_DIR=$Q/autotune
echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) uptime: $(uptime)"
bad=0
# SMOKE_ONLY: optional regex over "<kind>__<backend>" to smoke a subset.
for f in $(ls "$Q/todo" | grep -v rows | awk -F'__' '!seen[$2"__"$3]++' | grep -E "${SMOKE_ONLY:-.}"); do
  stem=${f%.jsonl}; kind=$(echo "$stem" | awk -F'__' '{print $2}'); backend=$(echo "$stem" | awk -F'__' '{print $3}')
  python3 "$ROOT/research/bench/kernels/smallest_spec.py" "$Q/todo/$stem.rows.jsonl" > "$OUT/$kind.$backend.spec.jsonl"
  t0=$(date +%s)
  # The NVFP4 MoE runner autotunes before timing; give it longer. run_call.sh kills only its own session.
  to=240; case "$kind" in nvfp4_fused_moe) to=420;; esac
  ( cd "$SIM" && PPM_KNAMES_DIR="$OUT/knames_$kind.$backend" "$ROOT/research/bench/kernels/run_call.sh" $to \
      "$OUT/$kind.$backend.json" "$OUT/$kind.$backend.err" \
      uv run --no-sync python "$ROOT/research/bench/kernels/ppimpl_profile.py" run "$kind" \
      --backend "$backend" --specs "$OUT/$kind.$backend.spec.jsonl" --fresh --gpu-name "NVIDIA B200" --json )
  rc=$?
  st=$(python3 -c "import json; d=json.load(open('$OUT/$kind.$backend.json')); r=d['results'][0]; print(r.get('status'), r.get('metrics',{}).get('time_ms'), (r.get('error') or '')[:200])" 2>/dev/null || echo "no-json")
  echo "$kind/$backend rc=$rc $(( $(date +%s) - t0 ))s: $st"
  case "$st" in ok*) ;; *) bad=1; tail -15 "$OUT/$kind.$backend.err";; esac
  [ $rc -eq 125 ] && { echo "GPU still held after cleanup; stopping"; break; }
done
echo "end $(date -Is) bad=$bad"
exit $bad
