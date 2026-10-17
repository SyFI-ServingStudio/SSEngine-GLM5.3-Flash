#!/bin/bash
# M6 (7_offloading.md, TIERS.md) runs on one 4-GPU server, into OUT (default measured/runs_m6). Usage, from the repo
# root (one run per job; the bench/run.py options follow the name):
#   mkdir -p measured/runs_m6/slurm
#   sbatch -p priority -J ppm6 --time=00:20:00 --mem=300G -o measured/runs_m6/slurm/%j.out bench/slurm/m6.sbatch \
#     tier_bringup --dram-gb 48 --ssd-gb none --run-name tier_bringup_dram
# SSD shard sets (ppimpl/kvpool/shards.py: set<k>/shard<q>.bin, 128 GB per shard) live in PPIMPL_SSD_DIR, default
# <main tree>/tmp/ssd_tier, kept across runs; the run takes the first free set. Create a set once, before the first
# SSD run: $PY -m ppimpl.kvpool.shards create --sets 1 --shards 4 --gb 128. df of the repo's file system is printed
# before and after.
# Memory guards: bench/slurm/memguard.py (start only with MemAvailable > pinned + 100 GB and memory PSI < 5%; a timed
# part that thrashes is stopped and marked INVALID.txt).
# HOP2=1 runs bench/tier_hop2.py (two-hop rates, GEMM and stage-edge interference) before the run.
#SBATCH --gres=gpu:4
set -uo pipefail
{   # one brace group: bash parses it whole before running, so editing this file mid-job cannot break the job
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m6}
mkdir -p "$R/slurm"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
export PPIMPL_SSD_DIR=${PPIMPL_SSD_DIR:-$MAIN/tmp/ssd_tier}
export PPIMPL_HANDOFF=${PPIMPL_HANDOFF:-$MAIN/third_party/handoff_v2_realistic}   # v2-realistic targets
trap 'rm -rf "$TMPDIR"; echo "== df after: $(df -h "$ROOT" | tail -1)"' EXIT
WHAT=${1:?run name (bench/run.py)}; shift
echo "== $(date -Is) gpus $G port $P: $WHAT $* (ssd dir $PPIMPL_SSD_DIR)"
echo "== df before: $(df -h "$ROOT" | tail -1)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
if [ -n "${HOP2:-}" ]; then                # HOP2=1: the two-hop probe (bench/tier_hop2.py) first; stop if it fails
  $PY -m bench.tier_hop2 --out "$R/tier_hop2/$JOB" ${HOP2_TESTS:+--tests $HOP2_TESTS} || { echo "== tier_hop2 failed"; exit 1; }
fi
# memory guards (bench/slurm/memguard.py): do not start on a host short of memory or thrashing; stop the timed part
# (marked INVALID.txt) if it starts thrashing
$PY bench/slurm/memguard.py precheck -- "$@" || { echo "== not started: host memory (see memguard line)"; exit 0; }
RUN=$WHAT; for ((i = 1; i <= $#; i++)); do [ "${!i}" = --run-name ] && { j=$((i + 1)); RUN=${!j}; }; done
setsid $PY -m bench.run "$WHAT" --inline --gpus "$G" --port "$P" --out-root "$R" "$@" &
RUNPID=$!
$PY bench/slurm/memguard.py watch --pid $RUNPID --run-dir "$R/$RUN" &
WD=$!
wait $RUNPID; rc=$?
kill $WD 2>/dev/null; wait $WD 2>/dev/null
[ -f "$R/$RUN/INVALID.txt" ] && { echo "== run invalid: $(cat "$R/$RUN/INVALID.txt")"; rc=4; }
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
}
