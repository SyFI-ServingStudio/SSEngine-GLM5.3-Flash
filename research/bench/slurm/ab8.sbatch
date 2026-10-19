#!/bin/bash
# Two 4-GPU runs at once in one 8-GPU job (an A/B pair on one node): run A on GPUs 0-3 with the CPUs of their NUMA
# node, run B on GPUs 4-7 with the other node's, each through benchmarks/jobs/m6.sbatch from its own run tree (code
# under test), with its own port, TMPDIR and SSD dir (AB_SSD_DIR: one dir for both, whose shard sets the two runs
# take one each). memguard prechecks/watches each run. Usage:
#   sbatch -p main -J ppab --gres=gpu:8 --mem=500G --time=00:35:00 -o <dir>/%j.out research/bench/slurm/ab8.sbatch \
#     TREE_A TREE_B -- <m6.sbatch args for A> -- <m6.sbatch args for B>
set -uo pipefail
M6=${SLURM_SUBMIT_DIR:?}/benchmarks/jobs/m6.sbatch   # submit from the lane: its m6.sbatch (SSENGINE_PORT) runs both
TA=$1 TB=$2; shift 2; [ "$1" = -- ] && shift
A=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do A+=("$1"); shift; done; shift; B=("$@")
G=${CUDA_VISIBLE_DEVICES:?}; IFS=, read -ra GS <<< "$G"; [ ${#GS[@]} = 8 ] || { echo "need 8 GPUs, got $G"; exit 1; }
GA=$(IFS=,; echo "${GS[*]:0:4}"); GB=$(IFS=,; echo "${GS[*]:4:4}")
node_cpus() {   # the job's CPUs on the NUMA node of the GPU with this PCI bus id
  local bus n; bus=$(nvidia-smi -i "$1" --query-gpu=pci.bus_id --format=csv,noheader | tr A-Z a-z)
  n=$(cat "/sys/bus/pci/devices/${bus:4}/numa_node"); python3 - "$n" <<'PY'
import os, sys
node = int(sys.argv[1]); out = []
def parse(s):
    r = []
    for p in s.strip().split(","):
        a, _, b = p.partition("-"); r += range(int(a), int(b or a) + 1)
    return r
mine = set(parse(open(f"/sys/devices/system/node/node{node}/cpulist").read())) & os.sched_getaffinity(0)
print(",".join(map(str, sorted(mine))))
PY
}
CA=$(node_cpus "${GS[0]}"); CB=$(node_cpus "${GS[4]}")
BASEP=$((20000 + ${SLURM_JOB_ID:-0} % 60 * 200))
echo "== ab8 job ${SLURM_JOB_ID:-local}: A $TA gpus $GA cpus $CA port $BASEP: ${A[*]}"
echo "== ab8 job ${SLURM_JOB_ID:-local}: B $TB gpus $GB cpus $CB port $((BASEP + 100)): ${B[*]}"
run() {   # tree gpus cpus port tag args...
  local tree=$1 g=$2 c=$3 p=$4 tag=$5; shift 5
  (cd "$tree" && SLURM_SUBMIT_DIR=$tree CUDA_VISIBLE_DEVICES=$g SSENGINE_PORT=$p \
     SSENGINE_TMP=${TMPDIR:-/tmp}/ab$tag SSENGINE_SSD_DIR=${AB_SSD_DIR:-$tree/../../tmp/ssd_tier_ab$tag} \
     taskset -c "$c" bash "$M6" "$@") > "${AB_LOG:-$SLURM_SUBMIT_DIR/measured/logs}/ab_${SLURM_JOB_ID:-local}_$tag.out" 2>&1
}
run "$TA" "$GA" "$CA" "$BASEP" A "${A[@]}" & PA=$!
run "$TB" "$GB" "$CB" "$((BASEP + 100))" B "${B[@]}" & PB=$!
wait $PA; ra=$?; wait $PB; rb=$?
echo "== ab8 done $(date -Is) A rc=$ra B rc=$rb"; tail -3 "${AB_LOG:-$SLURM_SUBMIT_DIR/measured/logs}"/ab_${SLURM_JOB_ID:-local}_{A,B}.out
exit $(( ra || rb ))
