#!/bin/bash
# Build SSEngine's environment from public sources (ENV.md has the versions and why each one).
#
#   env/make_env.sh [DIR]
#
# DIR (default: this repo) receives .venv/ and third_party/; lane worktrees under lanes/ share the main tree's.
# Run as the normal user, never as root. Needs git, curl, unzip, uv, and the CUDA 13 toolkit (nvcc) for
# causal-conv1d. The model weights are separate (README, "Reproduce").
#
# Overridable:
#   CUDA_HOME   CUDA 13 toolkit for the causal-conv1d build   default /usr/local/cuda
#   MAX_JOBS    parallel nvcc jobs for that build             default 16
#   TMPDIR      keep it short: vLLM's zmq ipc path is TMPDIR + "/" + a uuid and must fit in 107 characters
#   STEPS       the steps to run, in this order                default all of them:
#                 venv vllm conv1d   the serving environment (the Docker image runs these, install and jit:
#                                    docker/Dockerfile)
#                 install            this repo into the venv (editable, `ssengine` on the venv's PATH)
#                 jit                FlashInfer's JIT modules for sm_100a and the trtllm-gen cubins serving uses
#                                    (env/bake_jit.py; no GPU, ~5 min): no compile or download at the first start
#                 sim client corpus  the bench's handoff checkout, req-frontend and enwik9
#                 check              env/check_env.py
#   SSENGINE_ALLOW_ROOT=1   build as root: only inside a container build (docker/Dockerfile)
# Each step skips what is already in place, so a failed run can be repeated.
set -euo pipefail
[ "$(id -u)" != 0 ] || [ "${SSENGINE_ALLOW_ROOT:-}" = 1 ] || { echo "do not run as root"; exit 1; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=$(realpath "${1:-$HERE/..}")
TP=$DIR/third_party
PY=$DIR/.venv/bin/python
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MAX_JOBS=${MAX_JOBS:-16}
STEPS=${STEPS:-venv vllm conv1d install jit sim client corpus check}
mkdir -p "$TP/downloads"
want() { [[ " $STEPS " == *" $1 "* ]]; }
for s in $STEPS; do
  case $s in venv|vllm|conv1d|install|jit|sim|client|corpus|check) ;; *) echo "unknown step $s (STEPS)"; exit 1 ;; esac
done

fetch() {  # url sha256 file: download once, then always check the hash
  if [ ! -f "$3" ]; then curl -fL --retry 3 -o "$3.part" "$1"; mv "$3.part" "$3"; fi
  echo "$2  $3" | sha256sum -c --quiet -
}

# 1. Python 3.12 venv with the pinned packages: torch 2.13.0 (+cu130 from PyPI), FlashInfer nightly
#    0.7.2.dev20261008 and cutlass-dsl 4.8.0 instead of the fork's pins (ENV.md), by URL and sha256 in the lock.
if want venv; then
  [ -x "$PY" ] || uv venv --python 3.12 "$DIR/.venv"
  uv pip install --python "$PY" -r "$HERE/requirements.lock"
fi

# 2. The vLLM fork at 892da0822f (branch servingstudio-alignment), installed as its ALIGNMENT.md says: editable, with
#    the native extensions of the upstream wheel of its base commit 04730e8 (the fork changes no native code). The
#    wheel is fetched here and checked, then handed to the install.
if want vllm; then
  FORK_COMMIT=892da0822f7e866cec8e519727f014c45a5ad792
  BASE=04730e82700d0c9b957cda188dd1bbc8073e1fcd
  if [ ! -d "$TP/vllm/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/serendipity-zk/vllm.git "$TP/vllm"
  fi
  git -C "$TP/vllm" -c advice.detachedHead=false checkout "$FORK_COMMIT"
  WHL=$TP/downloads/vllm-0.30.1rc1.dev97+g04730e827-cp38-abi3-manylinux_2_28_x86_64.whl
  fetch "https://wheels.vllm.ai/$BASE/vllm-0.30.1rc1.dev97%2Bg04730e827-cp38-abi3-manylinux_2_28_x86_64.whl" \
    a6eb62ee6e2913a72a57dd41c870016c2f4d64e6546dda81ffbbd79b14a17f96 "$WHL"
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$BASE VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=$WHL uv pip install --python "$PY" --no-deps -e "$TP/vllm"
fi

# 3. Dao causal-conv1d 1.7.0 (the KDA short conv), built from its source tag against this torch: the released cu13
#    wheel is built for torch 2.10 and fails to load with 2.13.
if want conv1d; then
  CC=$TP/downloads/causal-conv1d-1.7.0.tar.gz
  # The wheel stays in third_party/wheels, so a rebuilt venv reinstalls it without compiling (~10 min).
  if ! compgen -G "$TP/wheels/causal_conv1d-1.7.0-*.whl" > /dev/null; then
    fetch https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.7.0.tar.gz \
      708bacd4fb4ef6cad3999c9360e2a0f9521f556748a745b60f746b275db73c09 "$CC"
    rm -rf "$TP/downloads/causal-conv1d-1.7.0" && tar -xzf "$CC" -C "$TP/downloads"
    PATH=$CUDA_HOME/bin:$PATH CAUSAL_CONV1D_FORCE_BUILD=TRUE uv build --python "$PY" --wheel --no-build-isolation \
      -o "$TP/wheels" "$TP/downloads/causal-conv1d-1.7.0"
  fi
  uv pip install --python "$PY" --no-deps "$TP"/wheels/causal_conv1d-1.7.0-*.whl
fi

# 3b. This repo, editable: the `ssengine` command (.venv/bin/ssengine) and `python -m ssengine` from any directory.
if want install; then
  uv pip install --python "$PY" --no-deps -e "$HERE/.."
fi

# 3c. FlashInfer's JIT modules (fp4 quantization, trtllm-gen FMHA, trtllm-gen fused MoE) built for sm_100a and the
#     trtllm-gen cubins they load, into FlashInfer's cache (~/.cache/flashinfer): what the first serving start would
#     otherwise compile (~10 min) and download. Built with /usr/bin/ninja, as the server's JIT checks (ENV.md).
if want jit; then
  PATH=/usr/bin:$PATH MAX_JOBS=$MAX_JOBS "$PY" "$HERE/bake_jit.py"
fi

# 4. The handoff and the client: ServingStudioSim at 820adda0 (branch pp-main: the handoff on top of its frozen commit
#    7bb224a6) with its req-frontend submodule (the profiler's vLLM/SGLang submodules are not needed); and a copy of
#    the profiler package at 7bb224a6, whose input builders and references the M1/M2 checks import.
if want sim; then
  SIM=$TP/ServingStudioSim
  [ -d "$SIM/.git" ] || git clone --filter=blob:none --no-checkout https://github.com/SyFI-ServingStudio/ServingStudioSim.git "$SIM"
  git -C "$SIM" -c advice.detachedHead=false checkout 820adda0b719e587b2bdc2ad95f446e11bee64db
  git -C "$SIM" submodule update --init alignment/load_generator/req-frontend
  if [ ! -d "$TP/sim_7bb224a6/profiling" ]; then
    mkdir -p "$TP/sim_7bb224a6"
    git -C "$SIM" archive 7bb224a699b88535bd71c6ead6db72a6967290e3 profiling | tar -x -C "$TP/sim_7bb224a6"
  fi
fi

# 4b. The client: req-frontend (request-factory) at 8723d90d (benchmarks.common REQ_FRONTEND_COMMIT: a session's slot is
#     freed when its last round completes, as in the sim). That commit is on a local branch (preserialize-body) until it
#     is pushed: REQ_FRONTEND_FROM names a checkout that has it (default: the GitHub repository).
if want client; then
  RF=$TP/req-frontend
  [ -d "$RF/.git" ] || git clone --no-checkout "${REQ_FRONTEND_FROM:-https://github.com/uw-syfi/request-factory.git}" "$RF"
  git -C "$RF" -c advice.detachedHead=false checkout 8723d90d4dcfcc526f49cce9fa7b50925f8635e0
fi

# 5. The prompt corpus: enwik9.
if want corpus; then
  if [ ! -f "$TP/enwik9" ]; then
    fetch http://mattmahoney.net/dc/enwik9.zip \
      99cdb5ac84392252d3f0912ccedd195bc95bd80cbef3b0cdf2eee4ad9a3b7a51 "$TP/downloads/enwik9.zip"
    unzip -o -q "$TP/downloads/enwik9.zip" -d "$TP"
  fi
  echo "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc  $TP/enwik9" | sha256sum -c --quiet -
fi

if want check; then
  "$PY" "$HERE/check_env.py"
fi
