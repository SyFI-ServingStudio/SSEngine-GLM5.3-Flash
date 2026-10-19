#!/bin/bash
# Smoke test of the Docker image (docker/Dockerfile, README "Docker") on the job's 4 GPUs: the image's default command
# (`<image> --served-model-name glm53-flash`: serve nvidia/GLM-5.3-Flash-NVFP4) comes up, /health, /v1/models, one
# /v1/completions (text), one with token ids and one /v1/chat/completions answer, then a graceful stop. From the repo
# root:
#   mkdir -p measured/logs
#   IMAGE=ppimpl sbatch -p priority --time=00:30:00 -o measured/logs/%j.out bench/slurm/docker_smoke.sbatch
# The container gets the job's GPUs by UUID (docker's devices are not Slurm's cgroup), the HF cache read-only
# (HF_HUB_OFFLINE) and the JIT caches in the volume CACHE (default ppimpl-cache-$USER). Options:
#   CACHE=fresh  a new empty cache volume, removed at the end: the first start of a pulled image
#   OFFLINE=1    no network in the container (--network none, FLASHINFER_NO_DOWNLOAD): requests through `docker exec`
#   SEED_OUT=F   after the run, write to F the cache files the run made that the image's seed lacks (the GPU seed,
#                docker/jit-seed/gpu.tar; run it with CACHE=fresh on an image of the `base` target)
#   TIERS=1      then start docker/compose.yaml (memlock, seccomp) with a DRAM and an SSD tier (TIER_ARGS, default
#                --dram-gb 20 --ssd-gb 8; shard set in SSD_DIR, default <main tree>/tmp/ssd_docker) and ask again
#   NATIVE=1     then serve the same model from this tree's environment and check it answers the same tokens
# Each start reports its JIT activity: FlashInfer module builds, CuTe-DSL and TileLang compiles, cubin downloads.
#SBATCH --gres=gpu:4
set -uo pipefail
{
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
IMAGE=${IMAGE:-ppimpl}
MODEL=${MODEL:-}
CACHE=${CACHE:-ppimpl-cache-$USER}
[ "$CACHE" = fresh ] && { CACHE=ppimpl-smoke-$JOB; FRESH=1; } || FRESH=0
P=${PPIMPL_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))}
NAME=ppimpl-smoke-$JOB
OUT=$ROOT/measured/logs/docker_smoke_$JOB
mkdir -p "$OUT"
# the job's GPUs: all nvidia-smi sees when Slurm's cgroup hides the others, else CUDA_VISIBLE_DEVICES of them
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"; nvidia-smi -L
SEL=(); [ "$(nvidia-smi -L | wc -l)" = 4 ] || SEL=(-i "${CUDA_VISIBLE_DEVICES:?}")
UUIDS=$(nvidia-smi "${SEL[@]}" --query-gpu=uuid --format=csv,noheader | paste -sd, -)
[ "$(tr , '\n' <<< "$UUIDS" | wc -l)" = 4 ] || { echo "== not 4 GPUs: $UUIDS"; exit 1; }
echo "== $(date -Is) image $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19)) gpus $UUIDS port $P" \
     "cache $CACHE fresh=$FRESH offline=${OFFLINE:-0}"
[ $FRESH = 1 ] && { docker volume rm "$CACHE" > /dev/null 2>&1; docker volume create "$CACHE" > /dev/null; }
COMPOSE=(docker compose -p "ppimpl-smoke-$JOB" -f "$ROOT/docker/compose.yaml" -f "$OUT/compose.gpus.yaml")
cleanup() {
  docker rm -f "$NAME" > /dev/null 2>&1
  [ -f "$OUT/compose.gpus.yaml" ] && "${COMPOSE[@]}" down -v > /dev/null 2>&1
  [ $FRESH = 1 ] && docker volume rm "$CACHE" > /dev/null 2>&1
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

# curl against the server: from the host, or inside the container when it has no network (EXEC=<container>)
EXEC=
req() { if [ -n "$EXEC" ]; then docker exec "$EXEC" curl "$@"; else curl "$@"; fi; }

wait_health() {   # url, running-check command; seconds to ready, or fail
  local t0=$SECONDS
  until req -sf "$1/health" > /dev/null; do
    eval "$2" || { echo "== server exited before it was ready"; return 1; }
    (( SECONDS - t0 > 1500 )) && { echo "== not ready after 1500 s"; return 1; }
    sleep 2
  done
  echo "== ready after $((SECONDS - t0)) s"
}

ask() {   # url, tag: the requests; answers to $OUT/<tag>_*.json
  local u=$1 tag=$2 rc=0
  req -sf "$u/v1/models" | tee "$OUT/${tag}_models.json"; echo
  req -sf "$u/v1/completions" -H 'Content-Type: application/json' -d \
    "{\"model\": \"glm53-flash\", \"prompt\": \"The capital of France is\", \"max_tokens\": 1}" \
    | tee "$OUT/${tag}_completion.json" || rc=1; echo
  req -sf "$u/v1/completions" -H 'Content-Type: application/json' -d \
    "{\"model\": \"glm53-flash\", \"prompt\": [$(seq -s, 1000 1 1999)], \"stream\": true, \"stream_options\": {\"include_usage\": true}}" \
    | tee "$OUT/${tag}_tokens.txt" || rc=1; echo
  req -sf "$u/v1/chat/completions" -H 'Content-Type: application/json' -d \
    '{"model": "glm53-flash", "messages": [{"role": "user", "content": "What is 2+2? Answer with one digit."}], "chat_template_kwargs": {"reasoning_effort": "low"}}' \
    | tee "$OUT/${tag}_chat.json" || rc=1; echo
  return $rc
}

jit_report() {   # run dir: the JIT activity in its worker logs
  local d=$1
  echo "== JIT in $(basename "$d"): FlashInfer builds $(cat "$d"/worker*.log | grep -c 'Building JIT module')," \
       "CuTe-DSL $(cat "$d"/worker*.log | grep -c 'Compiling CuTe-DSL kernel')," \
       "TileLang $(cat "$d"/worker*.log | grep -c 'TileLang begins to compile')," \
       "cubin downloads $(cat "$d"/worker*.log | grep -c 'cubin_loader.*Fetching')"
}

answer() { python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['choices'][0]; print(repr(c.get('text', c.get('message'))))" "$1"; }

NET=(-p "$P:8000"); URL=http://127.0.0.1:$P
if [ "${OFFLINE:-0}" = 1 ]; then NET=(--network none -e FLASHINFER_NO_DOWNLOAD=1); URL=http://127.0.0.1:8000; EXEC=$NAME; fi
ARGS=(--served-model-name glm53-flash); [ -n "$MODEL" ] && ARGS=(serve "$MODEL" "${ARGS[@]}")
docker run -d --name "$NAME" --gpus "\"device=$UUIDS\"" --ipc=host "${NET[@]}" \
  -e HF_HUB_OFFLINE=1 -v "$HF_HOME:/root/.cache/huggingface:ro" -v "$CACHE:/root/.cache" \
  "$IMAGE" "${ARGS[@]}" > /dev/null || exit 1
docker logs -f "$NAME" > "$OUT/docker.log" 2>&1 &
rc=0
if wait_health "$URL" "[ \"\$(docker inspect -f '{{.State.Running}}' $NAME)\" = true ]"; then
  ask "$URL" docker || rc=1
else
  rc=1
fi
t=$SECONDS; docker stop -t 120 "$NAME" > /dev/null
echo "== docker stop after $((SECONDS - t)) s, exit code $(docker inspect -f '{{.State.ExitCode}}' "$NAME")"
docker cp "$NAME:/workspace/run" "$OUT/docker_run" 2> /dev/null
docker rm "$NAME" > /dev/null
EXEC=
jit_report "$OUT/docker_run"
for f in completion chat; do echo "== $f: $(answer "$OUT/docker_$f.json")"; done

if [ -n "${SEED_OUT:-}" ] && [ $rc = 0 ]; then
  # every regular file the run added to the volume that the image's seed does not have, minus locks and scratch
  mkdir -p "$(dirname "$SEED_OUT")"
  docker run --rm --entrypoint bash -v "$CACHE:/c:ro" -v "$(realpath "$(dirname "$SEED_OUT")"):/out" "$IMAGE" -c '
    cd /c && find . -path ./huggingface -prune -o -type f -print | sed "s|^\./||" \
      | grep -Ev "(\.lock$|/tmp/|/\.staging/|^vllm/modelinfos/)" \
      | while read -r f; do [ -e "/opt/ppimpl/jit-seed/$f" ] || echo "$f"; done | sort > /out/.seed.list
    tar -cf "/out/'"$(basename "$SEED_OUT")"'" -T /out/.seed.list && rm /out/.seed.list' \
    && echo "== GPU seed $SEED_OUT: $(tar -tf "$SEED_OUT" | wc -l) files, $(du -h "$SEED_OUT" | cut -f1);" \
            "$(tar -tf "$SEED_OUT" | cut -d/ -f1 | sort | uniq -c | tr '\n' ' ')" \
    || { echo "== GPU seed export failed"; rc=1; }
fi

if [ "${TIERS:-0}" = 1 ] && [ $rc = 0 ]; then
  # docker/compose.yaml as a user runs it, on the job's GPUs and port, with a DRAM and an SSD tier
  SSD_DIR=${SSD_DIR:-$MAIN/tmp/ssd_docker}
  TIER_ARGS=${TIER_ARGS:---dram-gb 20 --ssd-gb 8}
  ids=$(tr , '\n' <<< "$UUIDS" | sed 's/.*/"&"/' | paste -sd, -)
  printf 'services:\n  ppimpl:\n    deploy:\n      resources:\n        reservations:\n          devices: !override\n            - {driver: nvidia, device_ids: [%s], capabilities: [gpu]}\n    environment: {HF_HUB_OFFLINE: "1"}\n' \
    "$ids" > "$OUT/compose.gpus.yaml"
  export PPIMPL_IMAGE=$IMAGE PPIMPL_PORT=$P HF_CACHE=$HF_HOME PPIMPL_SSD_DIR=$SSD_DIR \
         PPIMPL_ARGS="--served-model-name glm53-flash $TIER_ARGS --ssd-dir /ssd"
  echo "== compose: $PPIMPL_ARGS (ssd dir $SSD_DIR)"
  URL=http://127.0.0.1:$P      # compose publishes the port (its container has the network)
  "${COMPOSE[@]}" up -d || rc=1
  C=$("${COMPOSE[@]}" ps -q ppimpl)
  "${COMPOSE[@]}" logs -f > "$OUT/compose.log" 2>&1 &
  if [ $rc = 0 ] && wait_health "$URL" "[ \"\$(docker inspect -f '{{.State.Running}}' $C)\" = true ]"; then
    ask "$URL" tiers || rc=1
    for f in completion chat; do
      a=$(answer "$OUT/docker_$f.json"); b=$(answer "$OUT/tiers_$f.json")
      [ "$a" = "$b" ] && echo "== $f: tiers = HBM only: $a" || { echo "== $f: tiers $b != HBM only $a"; rc=1; }
    done
  else
    rc=1
  fi
  docker cp "$C:/workspace/run" "$OUT/tiers_run" 2> /dev/null
  "${COMPOSE[@]}" stop -t 120 > /dev/null
  echo "== compose exit code $(docker inspect -f '{{.State.ExitCode}}' "$C")"
  jit_report "$OUT/tiers_run"
  grep -h -E '^stage [0-9] (tiers|start-up):' "$OUT"/tiers_run/worker*.log
  "${COMPOSE[@]}" down -v > /dev/null 2>&1
fi

if [ "${NATIVE:-0}" = 1 ] && [ $rc = 0 ]; then
  HF_HUB_OFFLINE=1 $PY -m ppimpl serve ${MODEL:+"$MODEL"} --served-model-name glm53-flash --host 127.0.0.1 --port "$P" \
    --run-dir "$OUT/native_run" > "$OUT/native.log" 2>&1 &
  SRV=$!
  if wait_health "http://127.0.0.1:$P" "kill -0 $SRV 2> /dev/null"; then
    ask "http://127.0.0.1:$P" native || rc=1
  else
    rc=1
  fi
  kill -TERM $SRV; wait $SRV
  for f in completion chat; do
    a=$(answer "$OUT/docker_$f.json"); b=$(answer "$OUT/native_$f.json")
    [ "$a" = "$b" ] && echo "== $f: docker = native: $a" || { echo "== $f: docker $a != native $b"; rc=1; }
  done
fi
echo "== done $(date -Is) rc=$rc"
exit $rc
}
