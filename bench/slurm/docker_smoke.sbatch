#!/bin/bash
# Smoke test of the Docker image (docker/Dockerfile, README "Docker") on the job's 4 GPUs: `serve <model>` comes up,
# /health, /v1/models, one /v1/completions (text), one with token ids and one /v1/chat/completions answer, then a
# graceful stop. From the repo root:
#   mkdir -p measured/logs
#   IMAGE=ppimpl sbatch -p priority --time=00:30:00 -o measured/logs/%j.out bench/slurm/docker_smoke.sbatch
# NATIVE=1 then serves the same model from this tree's environment (`python -m ppimpl serve <model>`) and checks it
# answers the same tokens. The container gets the job's GPUs by UUID (docker's devices are not Slurm's cgroup), the
# HF cache read-only (HF_HUB_OFFLINE) and the JIT caches in the volume CACHE (default ppimpl-cache-$USER).
#SBATCH --gres=gpu:4
set -uo pipefail
{
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
IMAGE=${IMAGE:-ppimpl}
MODEL=${MODEL:-nvidia/GLM-5.3-Flash-NVFP4}
CACHE=${CACHE:-ppimpl-cache-$USER}
P=${PPIMPL_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))}
NAME=ppimpl-smoke-$JOB
OUT=$ROOT/measured/logs/docker_smoke_$JOB
mkdir -p "$OUT"
# the job's GPUs: all nvidia-smi sees when Slurm's cgroup hides the others, else CUDA_VISIBLE_DEVICES of them
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"; nvidia-smi -L
SEL=(); [ "$(nvidia-smi -L | wc -l)" = 4 ] || SEL=(-i "${CUDA_VISIBLE_DEVICES:?}")
UUIDS=$(nvidia-smi "${SEL[@]}" --query-gpu=uuid --format=csv,noheader | paste -sd, -)
[ "$(tr , '\n' <<< "$UUIDS" | wc -l)" = 4 ] || { echo "== not 4 GPUs: $UUIDS"; exit 1; }
echo "== $(date -Is) image $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19)) gpus $UUIDS port $P"
cleanup() { docker rm -f "$NAME" > /dev/null 2>&1; rm -rf "$TMPDIR"; }
trap cleanup EXIT

wait_health() {   # url, pid-or-container check command; seconds to ready, or fail
  local t0=$SECONDS
  until curl -sf "$1/health" > /dev/null; do
    eval "$2" || { echo "== server exited before it was ready"; return 1; }
    (( SECONDS - t0 > 1500 )) && { echo "== not ready after 1500 s"; return 1; }
    sleep 5
  done
  echo "== ready after $((SECONDS - t0)) s"
}

ask() {   # url, tag: the requests; answers to $OUT/<tag>_*.json
  local u=$1 tag=$2 rc=0
  curl -sf "$u/v1/models" | tee "$OUT/${tag}_models.json"; echo
  curl -sf "$u/v1/completions" -H 'Content-Type: application/json' -d \
    "{\"model\": \"glm53-flash\", \"prompt\": \"The capital of France is\", \"max_tokens\": 1}" \
    | tee "$OUT/${tag}_completion.json" || rc=1; echo
  curl -sf "$u/v1/completions" -H 'Content-Type: application/json' -d \
    "{\"model\": \"glm53-flash\", \"prompt\": [$(seq -s, 1000 1 1999)], \"stream\": true, \"stream_options\": {\"include_usage\": true}}" \
    | tee "$OUT/${tag}_tokens.txt" || rc=1; echo
  curl -sf "$u/v1/chat/completions" -H 'Content-Type: application/json' -d \
    '{"model": "glm53-flash", "messages": [{"role": "user", "content": "What is 2+2? Answer with one digit."}], "chat_template_kwargs": {"reasoning_effort": "low"}}' \
    | tee "$OUT/${tag}_chat.json" || rc=1; echo
  return $rc
}

docker run -d --name "$NAME" --gpus "\"device=$UUIDS\"" --ipc=host -p "$P:8000" \
  -e HF_HUB_OFFLINE=1 -v "$HF_HOME:/root/.cache/huggingface:ro" -v "$CACHE:/root/.cache" \
  "$IMAGE" serve "$MODEL" --served-model-name glm53-flash > /dev/null || exit 1
docker logs -f "$NAME" > "$OUT/docker.log" 2>&1 &
rc=0
if wait_health "http://127.0.0.1:$P" "[ \"\$(docker inspect -f '{{.State.Running}}' $NAME)\" = true ]"; then
  ask "http://127.0.0.1:$P" docker || rc=1
else
  rc=1
fi
t=$SECONDS; docker stop -t 120 "$NAME" > /dev/null
echo "== docker stop after $((SECONDS - t)) s, exit code $(docker inspect -f '{{.State.ExitCode}}' "$NAME")"
docker cp "$NAME:/workspace/run" "$OUT/docker_run" 2> /dev/null
docker rm "$NAME" > /dev/null

if [ "${NATIVE:-0}" = 1 ] && [ $rc = 0 ]; then
  HF_HUB_OFFLINE=1 $PY -m ppimpl serve "$MODEL" --served-model-name glm53-flash --host 127.0.0.1 --port "$P" \
    --run-dir "$OUT/native_run" > "$OUT/native.log" 2>&1 &
  SRV=$!
  if wait_health "http://127.0.0.1:$P" "kill -0 $SRV 2> /dev/null"; then
    ask "http://127.0.0.1:$P" native || rc=1
  else
    rc=1
  fi
  kill -TERM $SRV; wait $SRV
  for f in completion chat; do
    a=$(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); c=j['choices'][0]; print(repr(c.get('text', c.get('message'))))" "$OUT/docker_$f.json")
    b=$(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); c=j['choices'][0]; print(repr(c.get('text', c.get('message'))))" "$OUT/native_$f.json")
    [ "$a" = "$b" ] && echo "== $f: docker = native: $a" || { echo "== $f: docker $a != native $b"; rc=1; }
  done
fi
echo "== done $(date -Is) rc=$rc"
exit $rc
}
