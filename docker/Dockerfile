# syntax=docker/dockerfile:1
# pp-impl's serving image: GLM-5.3-Flash NVFP4 as a 4-stage pipeline on 4 x B200, with an OpenAI-compatible server
# (README "Docker"). From the repo root:
#
#   docker build -f docker/Dockerfile -t ppimpl .
#   docker compose -f docker/compose.yaml up       # or, HBM only:
#   docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -v ppimpl-cache:/root/.cache \
#     -p 8000:8000 ppimpl            # = serve nvidia/GLM-5.3-Flash-NVFP4; `ppimpl serve [<model>] [options]`
#
# The environment is env/make_env.sh's serving steps (STEPS="venv vllm conv1d install jit"), so ENV.md's versions
# hold here as on a host; the bench-only steps (ServingStudioSim, req-frontend, enwik9) are left out: serving imports
# none of them.
# The JIT kernels are in the image, so a fresh container starts without compiling or downloading (README "Docker"):
#   - FlashInfer's JIT modules and trtllm-gen cubins, built and fetched here without a GPU (make_env.sh step `jit`);
#   - what only a run on a B200 produces (Triton, TileLang, DeepGEMM, CuTe-DSL, inductor kernels): the GPU seed
#     docker/jit-seed/gpu.tar (not in git), made by bench/slurm/docker_smoke.sbatch with SEED_OUT from an image of the
#     `base` target. Optional: without it the image builds the same, and a fresh container's first start compiles
#     these (~5 min) into the cache volume.
# Both sit in /opt/ppimpl/jit-seed, laid out as /root/.cache; docker/entrypoint copies what is missing into
# /root/.cache at each start, so a mounted cache volume is seeded too. The CUDA toolkit stays: other shapes or a
# changed kernel still compile at run time.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS base

# parallel nvcc jobs of the causal-conv1d and FlashInfer JIT builds (make_env.sh's default; each takes a few GB of
# host memory)
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive
# git: the vLLM fork's checkout (check_env.py reads its HEAD); liburing2: the SSD tier's io_uring; libnuma1: the DRAM
# tier's NUMA placement; ninja-build:
# /usr/bin/ninja 1.11.1 for FlashInfer's JIT (ENV.md, "Other tools"); g++: ppimpl/native (kda_schedule.py)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates liburing2 libnuma1 ninja-build g++ \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /uvx /usr/local/bin/

# make_env.sh runs as root only here (PPIMPL_ALLOW_ROOT); uv's cache is a build cache mount, not an image layer
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python PPIMPL_ALLOW_ROOT=1 CUDA_HOME=/usr/local/cuda
WORKDIR /opt/ppimpl
COPY env/make_env.sh env/requirements.lock env/
RUN --mount=type=cache,target=/root/.cache/uv STEPS=venv env/make_env.sh
RUN --mount=type=cache,target=/root/.cache/uv STEPS=vllm env/make_env.sh && rm -rf third_party/downloads
RUN --mount=type=cache,target=/root/.cache/uv STEPS=conv1d MAX_JOBS=$MAX_JOBS env/make_env.sh \
 && rm -rf third_party/downloads third_party/wheels
# FlashInfer's JIT modules and cubins, built where a server builds them (/root/.cache/flashinfer: the paths are part of
# ninja's up-to-date check), then moved into the seed
COPY env/bake_jit.py env/
RUN STEPS=jit MAX_JOBS=$MAX_JOBS env/make_env.sh \
 && mkdir -p /opt/ppimpl/jit-seed && mv /root/.cache/flashinfer /opt/ppimpl/jit-seed/ && rm -rf /root/.cache

COPY pyproject.toml README.md ENV.md ./
COPY env/check_env.py env/
COPY ppimpl ppimpl
COPY docker/entrypoint docker/seccomp.json docker/
# check_env.py at build time, without a GPU: the native extensions link libcuda.so.1, which `--gpus` mounts at run
# time; here the toolkit's stub stands in (/opt/cuda-stub, used only by `check-env` without a driver: docker/entrypoint)
RUN --mount=type=cache,target=/root/.cache/uv STEPS=install env/make_env.sh \
 && mkdir -p /opt/cuda-stub && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stub/libcuda.so.1 \
 && LD_LIBRARY_PATH=/opt/cuda-stub .venv/bin/python env/check_env.py \
 && PPIMPL_NATIVE_DIR=/opt/ppimpl/jit-seed/ppimpl/native .venv/bin/python -c \
    'from ppimpl.model.kda_schedule import _load; _load()'

# Every JIT and autotune cache under /root/.cache (one volume keeps them across containers); the weights in the HF
# cache at /root/.cache/huggingface (mount the host's); the run logs in /workspace/run (`--run-dir`).
# the venv last on PATH: `python`, `ppimpl` from it (the image has no other Python), `ninja` the system's
ENV PATH=$PATH:/opt/ppimpl/.venv/bin \
    HF_HOME=/root/.cache/huggingface \
    TRITON_CACHE_DIR=/root/.cache/triton \
    TILELANG_CACHE_DIR=/root/.cache/tilelang \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    TMPDIR=/tmp
WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["/opt/ppimpl/docker/entrypoint"]
CMD ["serve"]

# the GPU seed, when the build context has it (bench/slurm/docker_smoke.sbatch SEED_OUT=docker/jit-seed/gpu.tar on an
# image of the `base` target); the bind mounts docker/, which always exists, so a checkout without the seed builds too
FROM base
RUN --mount=type=bind,source=docker,target=/tmp/ctx-docker \
    if [ -f /tmp/ctx-docker/jit-seed/gpu.tar ]; then tar -xf /tmp/ctx-docker/jit-seed/gpu.tar -C /opt/ppimpl/jit-seed \
      && echo "GPU seed: $(tar -tf /tmp/ctx-docker/jit-seed/gpu.tar | wc -l) files"; \
    else echo "no docker/jit-seed/gpu.tar: the image has no GPU seed (the first start compiles those kernels)"; fi
