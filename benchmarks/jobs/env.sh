# Sourced by every job script here. Submit from the repo root (sbatch copies the script, so a job finds this file
# through the submit directory); `bash research/bench/slurm/<x>.sbatch` finds it from the script's own path.
#
# Every path is a variable with a default. sbatch passes the submitting shell's environment, so export an override
# before submitting:
#   SSENGINE_PYTHON   the env's interpreter                 default <main tree>/.venv/bin/python
#   HF_HOME         Hugging Face cache with the weights   default ~/.cache/huggingface (SSENGINE_CHECKPOINT: ssengine/config.py)
#   SSENGINE_TMP      parent of the per-job TMPDIR          default $TMPDIR, else /tmp
#   SSENGINE_SIM, SSENGINE_ENWIK9: the handoff/req-frontend checkout and the corpus (benchmarks/common.py)
# <main tree> is the repo, or for a lane worktree under lanes/ the tree that holds .venv and third_party/.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAIN=$ROOT
[ "$(basename "$(dirname "$ROOT")")" = lanes ] && MAIN=$(dirname "$(dirname "$ROOT")")
cd "$ROOT" || exit 1
PY=${SSENGINE_PYTHON:-$MAIN/.venv/bin/python}
JOB=${SLURM_JOB_ID:-local}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# One TMPDIR per job, short: vLLM's zmq ipc path is TMPDIR + "/" + a uuid and must fit in 107 characters.
export TMPDIR=${SSENGINE_TMP:-${TMPDIR:-/tmp}}/pp$JOB
if (( ${#TMPDIR} > 64 )); then echo "TMPDIR $TMPDIR is over 64 characters (vLLM's zmq ipc path); set SSENGINE_TMP"; exit 1; fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "job $JOB host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} $(git -C "$ROOT" log --oneline -1)"
uptime
