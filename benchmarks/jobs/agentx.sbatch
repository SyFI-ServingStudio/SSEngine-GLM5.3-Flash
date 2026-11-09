#!/bin/bash
# One AgentX run (benchmarks/agentx.py): server and AgentX client in one job. --gres must match the layout's GPU count
# (tp x pp x dp). From the repo root:
#   mkdir -p runs/agentx-slurm      # job logs outside runs/agentx/, which holds only run directories
#   sbatch -p <partition> --gres=gpu:4 --time=02:30:00 --mem=400G -o runs/agentx-slurm/%j.out \
#       benchmarks/jobs/agentx.sbatch --engine vllm --layout dp4-ep --concurrency 128
#   ... --mem=600G ... --engine ssengine --layout pp4 --concurrency 128 --host-kv-gb 100 --ssd-gb 512 --ssd-dir <shards>
#   ... --time=00:45:00 ... --engine sglang --layout tp4 --concurrency 8 --smoke
# Arguments go to `python -m benchmarks.agentx run`. Time: model load (SSEngine ~2 min, vLLM ~6 min, SGLang ~12 min;
# SGLang pp4 with an empty AGENTX_CACHE_DIR ~36 min of MoE tuning) + dataset load and warm-up (our runs: c128 5-12 min,
# c512 at 1/lane 12-32 min; the first run also downloads the dataset) + --duration (1800 s; smoke 300 s). Memory:
# 4 x --host-kv-gb GB of KV + up to ~130 GiB for the engine (peak ~500 GiB at N = 100: 600G). docs/eval/agentx.md,
# "Budget".
set -uo pipefail
{   # one brace group: bash parses it whole before running, so editing this file mid-job cannot break the job
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/benchmarks/jobs/env.sh"
# /usr/bin/ninja first (FlashInfer JIT, notes.md), but /usr/local/bin before it: cayenne's docker there is the wrapper
# that runs containers in the job's rootless Podman with its GPUs and /scratch/hf; /usr/bin/docker bypasses it
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=/usr/local/bin:/usr/bin:$PATH
# Engine images' JIT/autotune caches: regenerable, so on /scratch (cayenne: $HOME is for what is worth keeping)
export AGENTX_CACHE_DIR=${AGENTX_CACHE_DIR:-/scratch/$USER/agentx-engines}
# not exec: env.sh's EXIT trap removes the job's TMPDIR when this shell exits
"$PY" -m benchmarks.agentx run --python "$PY" "$@"
exit $?
}
