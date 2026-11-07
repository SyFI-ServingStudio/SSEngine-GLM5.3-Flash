#!/bin/bash
# One AgentX run (benchmarks/agentx.py): server and AgentX client in one job. --gres must match the layout's GPU count
# (tp x pp x dp). From the repo root:
#   mkdir -p runs/agentx/slurm
#   sbatch -p <partition> --gres=gpu:4 --time=01:00:00 --mem=400G -o runs/agentx/slurm/%j.out \
#       benchmarks/jobs/agentx.sbatch --engine vllm --layout dp4-ep --concurrency 32
#   ... --engine ssengine --layout pp4 --concurrency 32 --server-arg=--dram-gb --server-arg=50
#   ... --engine sglang --layout tp4-ep --concurrency 32 --smoke
# Arguments go to `python -m benchmarks.agentx run`. Time: model load (ssengine ~1 min, vLLM/SGLang 5-15 min cold) +
# AgentX's dataset load and warm-up (~5 min first time, cached in ~/.cache/aiperf) + --duration (1800 s; smoke 300 s).
set -uo pipefail
{   # one brace group: bash parses it whole before running, so editing this file mid-job cannot break the job
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/benchmarks/jobs/env.sh"
# /usr/bin/ninja first (FlashInfer JIT, notes.md), but /usr/local/bin before it: cayenne's docker there is the wrapper
# that runs containers in the job's rootless Podman with its GPUs and /scratch/hf; /usr/bin/docker bypasses it
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=/usr/local/bin:/usr/bin:$PATH
# Engine images' JIT/autotune caches: regenerable, so on /scratch (cayenne: $HOME is for what is worth keeping)
export AGENTX_CACHE_DIR=${AGENTX_CACHE_DIR:-/scratch/$USER/agentx-engines}
exec "$PY" -m benchmarks.agentx run --python "$PY" "$@"
}
