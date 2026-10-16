# Sourced by every job script here. Submit from the repo root (sbatch copies the script, so a job finds this file
# through the submit directory); `bash bench/slurm/<x>.sbatch` finds it from the script's own path.
#
# Every path is a variable with a default. sbatch passes the submitting shell's environment, so export an override
# before submitting:
#   PPIMPL_PYTHON   the env's interpreter                 default <main tree>/.venv/bin/python
#   HF_HOME         Hugging Face cache with the weights   default ~/.cache/huggingface (PPIMPL_CHECKPOINT: ppimpl/config.py)
#   PPIMPL_TMP      parent of the per-job TMPDIR          default $TMPDIR, else /tmp
#   PPIMPL_SIM, PPIMPL_ENWIK9: the handoff/req-frontend checkout and the corpus (bench/common.py)
# <main tree> is the repo, or for a lane worktree under lanes/ the tree that holds .venv and third_party/.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MAIN=$ROOT
[ "$(basename "$(dirname "$ROOT")")" = lanes ] && MAIN=$(dirname "$(dirname "$ROOT")")
cd "$ROOT" || exit 1
PY=${PPIMPL_PYTHON:-$MAIN/.venv/bin/python}
JOB=${SLURM_JOB_ID:-local}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# One TMPDIR per job, short: vLLM's zmq ipc path is TMPDIR + "/" + a uuid and must fit in 107 characters.
export TMPDIR=${PPIMPL_TMP:-${TMPDIR:-/tmp}}/pp$JOB
if (( ${#TMPDIR} > 64 )); then echo "TMPDIR $TMPDIR is over 64 characters (vLLM's zmq ipc path); set PPIMPL_TMP"; exit 1; fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
# Dao causal-conv1d 1.7.0, built from source for this torch (ENV.md)
export PYTHONPATH=$MAIN/third_party/causal_conv1d_try/pkg${PYTHONPATH:+:$PYTHONPATH}
echo "job $JOB host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} $(git -C "$ROOT" log --oneline -1)"
uptime
