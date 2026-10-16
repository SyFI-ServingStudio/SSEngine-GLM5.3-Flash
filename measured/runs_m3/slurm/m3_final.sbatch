#!/bin/bash
#SBATCH -J ppimpl-m3-final
#SBATCH --gres=gpu:8
#SBATCH --time=00:50:00
#SBATCH -p main
#SBATCH -o /raid/kanzhu/ServingStudio/pp-impl/measured/runs_m3/slurm/%j.out
# Final M3 run on the main tree's defaults (CE transport, intake_procs 4, CUDA graphs): every one_<L> and every
# saturated_<L>, 1k..128k. Two 4-GPU servers side by side: half A = GPUs 0-3 on NUMA node 0's CPUs, half B = GPUs 4-7
# on node 1's. A runs saturated 1k..32k while B runs one_1k..one_128k; then saturated_64k and _128k run on A alone
# (B idle). Each run.json records half, gpus and run_cpus. Ends with measured/iteration.md (bench/report.py).
# DRY=1 bash m3_final.sbatch: print the plan only.
set -uo pipefail
set -a; source /raid/kanzhu/ServingStudio/.env; set +a
export TMPDIR=/raid/tmp/kanzhu/ppr${SLURM_JOB_ID:-dry}; mkdir -p "$TMPDIR"
export HF_HOME=/raid/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/raid/kanzhu/ServingStudio/pp-impl/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
cd ${TREE:-/raid/kanzhu/ServingStudio/pp-impl}
PY=/raid/kanzhu/ServingStudio/pp-impl/.venv/bin/python
M=$PWD/measured
R=$M/runs_m3
JOB=${SLURM_JOB_ID:-dry}
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GA="${G[0]},${G[1]},${G[2]},${G[3]}"; GB="${G[4]},${G[5]},${G[6]},${G[7]}"
ALLOWED=$($PY -c "import os;print(' '.join(map(str,sorted(os.sched_getaffinity(0)))))")
cpus_of_node() { $PY -c "
a=set(map(int,'$ALLOWED'.split()));n=open('/sys/devices/system/node/node$1/cpulist').read().strip()
c=set()
for p in n.split(','):
    x,_,y=p.partition('-'); c|=set(range(int(x),int(y or x)+1))
print(','.join(map(str,sorted(a&c))))"; }
CA=$(cpus_of_node 0); CB=$(cpus_of_node 1)
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
echo "== $(date -Is) $(git log --oneline -1) host $(hostname) gpus A=$GA B=$GB ports A=$P B=$((P+50))"
echo "   cpus A=$CA"; echo "   cpus B=$CB"; uptime
$PY -c "from ppimpl.config import RunConfig as C; c=C(); print('   defaults:', {k: getattr(c, k, None) for k in ('intake_procs', 'cuda_graphs', 'max_in_flight')})"
[ -n "${DRY:-}" ] || nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader

half() {   # half <A|B> <run names...>: one bench/run.py per name (fresh ports per call), into $R
  local h=$1 r=0; shift
  local g=$GA c=$CA p=$P; [ "$h" = B ] && { g=$GB; c=$CB; p=$((P+50)); }
  for n in "$@"; do
    echo "== $(date -Is) half $h start $n"
    if [ -n "${DRY:-}" ]; then echo "   PPIMPL_HALF=$h taskset -c <$h cpus> bench/run.py $n --gpus $g --port $p --out-root $R"; continue; fi
    PPIMPL_HALF=$h taskset -c "$c" $PY -m bench.run "$n" --inline --client-timeout 900 --gpus "$g" --port "$p" \
      --out-root "$R" >> "$R/slurm/${JOB}_half_$h.log" 2>&1 || r=1
    echo "== $(date -Is) half $h done $n rc=$r"
    # first run of the half failed: the defaults are broken (server start, graphs); stop instead of holding 8 GPUs
    [ $r = 1 ] && [ "$n" = "$1" ] && { echo "== half $h: first run failed, stopping this half"; return 1; }
  done
  return $r
}
rc=0
half A saturated_1k saturated_2k saturated_4k saturated_8k saturated_16k saturated_32k & a=$!
half B one_1k one_2k one_4k one_8k one_16k one_32k one_64k one_128k & b=$!
wait $a || rc=1; wait $b || rc=1; uptime
half A saturated_64k saturated_128k || rc=1     # B idle: the quiet-host check
uptime
[ -n "${DRY:-}" ] || $PY -m bench.report --summarize "$R" --out "$M"
grep -h "FAILED\| -> " "$R/slurm/${JOB}_half_"*.log 2>/dev/null
echo "== done $(date -Is) rc=$rc"
rm -rf "$TMPDIR"
exit $rc
