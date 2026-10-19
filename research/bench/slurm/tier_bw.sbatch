#!/bin/bash
# Host <-> GPU transfer bandwidth for the DRAM tier design (research/bench/tier_bw.py), plus the GPUDirect Storage check.
# Usage, from the repo root (OUT default research/measured/tier_bw):
#   mkdir -p research/measured/tier_bw/slurm
#   sbatch -p main --nice=0 --gres=gpu:4 --mem=64G --time=00:15:00 -o research/measured/tier_bw/slurm/%j.out research/bench/slurm/tier_bw.sbatch
#SBATCH -J ssengine-tierbw
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
R=$ROOT/${OUT:-measured/tier_bw}/$JOB
mkdir -p "$R"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,memory.used --format=csv,noheader
nvidia-smi topo -m > "$R/topo.txt" 2>&1
$PY -m research.bench.tier_bw --out "$R" ${TESTS:+--tests $TESTS}; rc=$?
GDS=${CUDA_HOME:-/usr/local/cuda}/gds/tools
[ -x "$GDS/gdscheck" ] && "$GDS/gdscheck" -p > "$R/gdscheck.txt" 2>&1
echo "== done $(date -Is) rc=$rc"; uptime
exit $rc
