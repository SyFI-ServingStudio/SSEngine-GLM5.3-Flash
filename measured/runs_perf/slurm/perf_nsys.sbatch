#!/bin/bash
#SBATCH -J ppimpl-perf-nsys
#SBATCH --gres=gpu:4
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/lanes/perf/measured/runs_perf/slurm/%j.out
# Why is the same 8192-token microbatch ~51 ms per stage alone and 55-58 ms in a saturated run's steady state?
# Runs from lanes/perf (branch lane-perf), one server each, under Nsight Systems unless noted:
#   early/saturated_8k    nsys window mbs 0-9 (the ramp)
#   steady/saturated_8k   nsys window mbs 1000-1009
#   one/one_8k            nsys, all 20 microbatches (isolated)
#   sync/saturated_8k     no nsys, PPIMPL_SYNC_EACH_MB=1 (host waits for each microbatch: no run-ahead), A/B for (c)
# bench/nsys_trace.py attributes kernels; measured/runs_perf/kclass.py compares kernel classes.
#   sbatch -p priority --time=00:28:00 measured/runs_perf/slurm/perf_nsys.sbatch
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppf${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
export PPIMPL_REQ_FRONTEND_BIN=/raid/kanzhu/ServingStudio/pp-impl/build/req-frontend/release/session_runner
cd /raid/kanzhu/ServingStudio/pp-impl/lanes/perf
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
R=$PWD/measured/runs_perf
JOB=${SLURM_JOB_ID:-dry}
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) host $(hostname) gpus $G port $P"; uptime
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
rc=0
run() {   # run <tag> <port offset> <run.py args...>
  local tag=$1 off=$2; shift 2
  echo "== $(date -Is) start $tag: $*"; uptime
  $PY bench/run.py "$@" --inline --client-timeout 900 --gpus "$G" --port $((P + off)) --out-root "$R/$tag" \
    >> "$R/slurm/${JOB}_$tag.log" 2>&1 || rc=1
  echo "== $(date -Is) done $tag rc=$rc"; uptime
}
run early 0 saturated_8k --nsys --nsys-mbs 0:10
run steady 20 saturated_8k --nsys --nsys-mbs 1000:1010
run one 40 one_8k --nsys
export PPIMPL_SYNC_EACH_MB=1
run sync 60 saturated_8k
grep -h "FAILED\| -> " "$R"/slurm/${JOB}_*.log 2>/dev/null
echo "== done $(date -Is) rc=$rc"
rm -rf "$TMPDIR"
exit $rc
