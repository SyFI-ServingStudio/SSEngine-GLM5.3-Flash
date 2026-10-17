#!/bin/bash
# M6 (7_offloading.md, TIERS.md) runs on one 4-GPU server, into OUT (default measured/runs_m6). Usage, from the repo
# root (one run per job; the bench/run.py options follow the name):
#   mkdir -p measured/runs_m6/slurm
#   sbatch -p priority -J ppm6 --time=00:20:00 --mem=300G -o measured/runs_m6/slurm/%j.out bench/slurm/m6.sbatch \
#     tier_bringup --dram-gb 48 --ssd-gb none --run-name tier_bringup_dram
# SSD files (one per GPU, TierConfig.ssd_dir) go to PPIMPL_SSD_DIR, default <repo>/tmp/ssd_<job>; the directory is
# deleted when the job ends, and df of the repo's file system is printed before and after.
# HOP2=1 runs bench/tier_hop2.py (two-hop rates, GEMM and stage-edge interference) before the run.
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m6}
mkdir -p "$R/slurm"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
export PPIMPL_SSD_DIR=${PPIMPL_SSD_DIR:-$ROOT/tmp/ssd_$JOB}
export PPIMPL_HANDOFF=${PPIMPL_HANDOFF:-$MAIN/third_party/handoff_v2_realistic}   # v2-realistic targets
trap 'rm -rf "$PPIMPL_SSD_DIR" "$TMPDIR"; echo "== df after: $(df -h "$ROOT" | tail -1)"' EXIT
WHAT=${1:?run name (bench/run.py)}; shift
echo "== $(date -Is) gpus $G port $P: $WHAT $* (ssd dir $PPIMPL_SSD_DIR)"
echo "== df before: $(df -h "$ROOT" | tail -1)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
if [ -n "${HOP2:-}" ]; then                # HOP2=1: the two-hop probe (bench/tier_hop2.py) first; stop if it fails
  $PY -m bench.tier_hop2 --out "$R/tier_hop2/$JOB" || { echo "== tier_hop2 failed"; exit 1; }
fi
$PY -m bench.run "$WHAT" --inline --gpus "$G" --port "$P" --out-root "$R" "$@"; rc=$?
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
