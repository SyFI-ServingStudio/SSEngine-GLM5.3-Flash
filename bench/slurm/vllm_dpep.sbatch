#!/bin/bash
# vLLM DP4 + EP4 baseline on the M6 session workload (sessions_2ps, trace-timed, --max-concurrency N, 22 min), one
# engine per GPU, MoE experts sharded over the 4. Server and client in one job. Usage, from the repo root:
#   mkdir -p measured/vllm_dpep/slurm
#   N=400 sbatch -p main --time=00:50:00 --mem=250G -J vdpep -o measured/vllm_dpep/slurm/%j.out \
#       bench/slurm/vllm_dpep.sbatch                       # A: HBM only
#   N=400 OFFLOAD_GB=50 sbatch ... --mem=450G ...         # B: native CPU offload, 50 GB (1e9 B) per GPU
#   SMOKE=1 N=8 sbatch -p priority --time=00:30:00 ...    # smoke: 3 min of client
# Variables: N (client --max-concurrency), TRACE (default the v3 trace data/tracelab-v3/m6/sessions_2ps_v3.csv), OFFLOAD_GB (per GPU; --kv-offloading-size is per DP engine in GiB, since
# CPUOffloadingSpec divides by the engine's world size = TP x PP = 1), CLIENT_TIMEOUT (default 22m; SMOKE 3m),
# OUT (default <main tree>/measured/vllm_dpep), RUN (run name), VLLM_EXTRA (extra server args),
# POLICY (vllm-router policy in front of the server, default cache_aware = prefix-aware; none = no router, the client
# talks to vLLM's own load-only DP balancer as in job 8199), MNBT (--max-num-batched-tokens, default 32768 = our
# engine's mb_tokens).
# Router: vllm-router 0.1.15 (PyPI, github.com/vllm-project/router) in <main tree>/tmp/vllm-router-venv
# (uv venv --python 3.12 && uv pip install vllm-router==0.1.15). --intra-node-data-parallel-size 4 expands the one vLLM
# URL into 4 DP-aware workers and sends X-data-parallel-rank, which vLLM's DPLBAsyncMPClient honors (the request's
# data_parallel_rank bypasses its load scoring). cache_aware hashes the request's token ids (joined as text) into a
# radix tree per worker: prefix match >= cache_threshold 0.3 -> that worker, else the worker with the smallest tree;
# shortest queue instead when max - min in-flight load > 64 and max > 1.5 x min.
# --max-tree-size 200000000 characters per worker (default 67108864): token ids are ~6.5 characters each, so the
# default tree tracks only ~10M tokens per engine, less than one engine's HBM KV (13.4M) and DRAM tier (~10M); an
# evicted session would be routed by load to a rank without its cache.
# Memory: refuses to start unless MemAvailable > need + 100 GB (need = 150 GB + 4 x OFFLOAD_GB); a watchdog stops the
# job (INVALID.txt) when MemAvailable < 40 GB or memory PSI full avg10 > 20%. oom_score_adj 1000 on everything here.
#SBATCH --gres=gpu:4
set -uo pipefail
{   # one brace group: bash parses it whole before running, so editing this file mid-job cannot break the job
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
echo 1000 > /proc/self/oom_score_adj
export CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=/usr/bin:$PATH   # /usr/bin/ninja first (FlashInfer JIT, notes.md)
export HF_HOME=/raid/hf
CKPT=/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348
H=/raid/kanzhu/ServingStudio/wt-pp-hitrule/doc/handoff/glm53_flash_pp4_b200_v2/data
# The M6 runs' req-frontend (8723d90) with one change, bench/vllm_dpep_client_preflight.patch: the prefix-cache
# preflight probe is 32768 tokens, not 8192. vLLM's hybrid KV for this model at TP1 has an 8576-token block (attention
# page padded to the KDA state page), so an 8192-token probe can never hit and the unpatched client refuses to start
# (job 8191). The workload path is unchanged. Build: clone third_party/req-frontend at 8723d90, apply the patch,
# cargo build --release --features runtime --bin session_runner (CARGO_TARGET_DIR below).
CLIENT=${CLIENT:-$MAIN/tmp/vllm_dpep_client/build/release/session_runner}
TRACE=${TRACE:-/raid/kanzhu/ServingStudio/data/tracelab-v3/m6/sessions_2ps_v3.csv}   # v3 new-model trace (A/B)
N=${N:?N (client max concurrency)}
OFF=${OFFLOAD_GB:-}
POLICY=${POLICY:-cache_aware}
MNBT=${MNBT:-32768}
TAG=$([ -n "$OFF" ] && echo "off${OFF}" || echo hbm)
[ "$POLICY" != none ] && TAG=${TAG}_${POLICY}
[ "$MNBT" != 32768 ] && TAG=${TAG}_mnbt${MNBT}
[ -n "${SMOKE:-}" ] && TAG=smoke_$TAG
RUN=${RUN:-${TAG}_n${N}}_$JOB
RD=${OUT:-$MAIN/measured/vllm_dpep}/$RUN
mkdir -p "$RD/prom"
P=$((20000 + JOB % 120 * 100 + 50))           # job-derived, below 32768 (notes.md); DP RPC port P+10
for p in $P $((P + 10)) $((P + 20)) $((P + 30)); do   # vLLM, DP RPC, router, router metrics
  ss -ltnH "sport = :$p" | grep -q . && { echo "== port $p in use, not started"; exit 1; }
done

# --- memory precheck (per NUMA node printed; no binding anywhere, the kernel places pages)
NEED=$((150 + 4 * ${OFF:-0}))
avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
echo "== memory: MemAvailable ${avail} GiB, need ${NEED} + 100; $(numactl -H | grep free | tr '\n' ';')"
echo "== PSI $(head -2 /proc/pressure/memory | tr '\n' ' '); /dev/shm $(df -h /dev/shm | tail -1)"
(( avail > NEED + 100 )) || { echo "== not started: host memory"; exit 0; }

# --- server
ARGS=(serve "$CKPT" --served-model-name glm53-flash-pp4 --host 127.0.0.1 --port $P
      --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel --data-parallel-rpc-port $((P + 10))
      --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --mamba-cache-mode align
      --max-model-len 1048576 --language-model-only --tokens-only --enable-prompt-tokens-details
      --max-num-seqs 1016 --max-num-batched-tokens $MNBT)
# --max-num-seqs 1016, not the default 1024: the fork's KDA decode kernel (fused_recurrent_kda) launches grid
# z = seqs x 64 KDA heads (TP1), and 1024 x 64 = 65536 exceeds CUDA's 65535 limit (job 8189: Triton "invalid argument"
# during the CUDA-graph memory profile at capture size 1024). 1016 is the largest multiple of 8 below 1024.
[ -n "$OFF" ] && ARGS+=(--kv-offloading-size "$(python3 -c "print(round($OFF * 1e9 / 2**30, 4))")"
                        --kv-offloading-backend native)
ARGS+=(${VLLM_EXTRA:-})
echo "$PY -m vllm.entrypoints.cli.main ${ARGS[*]}" | tee "$RD/server_command.txt"
git -C "$MAIN/third_party/vllm" log --oneline -1 > "$RD/vllm_commit.txt"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader | tee "$RD/gpus_before.txt"
uptime > "$RD/uptime.txt"
T0=$(date +%s)
setsid $PY -m vllm.entrypoints.cli.main "${ARGS[@]}" > "$RD/server.log" 2>&1 &
SRV=$!
stop_server() { kill -INT -- -$SRV 2>/dev/null; for _ in $(seq 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
                kill -KILL -- -$SRV 2>/dev/null; }
trap 'stop_server; rm -rf "$TMPDIR"' EXIT

# --- watchdog (host memory)
( while sleep 5; do
    a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    f=$(awk '/^full/ {split($2, x, "="); print int(x[2])}' /proc/pressure/memory)
    if (( a < 40 || f > 20 )); then
      echo "host memory: MemAvailable ${a} GiB, PSI full avg10 ${f}% at $(date -Is)" > "$RD/INVALID.txt"
      kill -INT -- -$SRV; pkill -INT -f "$RD/requests.jsonl"; exit; fi
  done ) &
WD=$!

until curl -sf "http://127.0.0.1:$P/health" > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "== server exited during start-up"; tail -40 "$RD/server.log"; kill $WD; exit 3; }
  (( $(date +%s) - T0 > 2700 )) && { echo "== server not ready in 45 min"; kill $WD; exit 3; }
  sleep 5
done
echo "== ready after $(( $(date +%s) - T0 )) s" | tee "$RD/ready.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$RD/gpus_ready.txt"
numactl -H | grep free > "$RD/numa_ready.txt"
curl -s "http://127.0.0.1:$P/metrics" > "$RD/prom/ready.prom"
URL=http://127.0.0.1:$P
if [ "$POLICY" != none ]; then
  RCMD=($MAIN/tmp/vllm-router-venv/bin/vllm-router --host 127.0.0.1 --port $((P + 20)) --worker-urls "$URL"
        --intra-node-data-parallel-size 4 --policy "$POLICY" --max-tree-size 200000000
        --prometheus-host 127.0.0.1 --prometheus-port $((P + 30))
        --log-dir "$RD/router" --log-level info)
  echo "${RCMD[*]}" > "$RD/router_command.txt"
  "${RCMD[@]}" > "$RD/router.log" 2>&1 &
  RT=$!
  trap 'kill $RT 2>/dev/null; stop_server; rm -rf "$TMPDIR"' EXIT
  URL=http://127.0.0.1:$((P + 20))
  for _ in $(seq 120); do curl -sf "$URL/health" > /dev/null && break; sleep 1; done
  curl -sf "$URL/health" > /dev/null || { echo "== router not up"; tail -20 "$RD/router.log"; kill $WD; exit 3; }
  echo "== router up: ${RCMD[*]}"
fi

# --- client: exactly the M6 command (client_command.txt of runs_m6/st_tiers_n400) but base URL and paths
CT=${CLIENT_TIMEOUT:-$([ -n "${SMOKE:-}" ] && echo 3m || echo 22m)}
CMD=(timeout "$CT" "$CLIENT" --trace "$TRACE" --input-file-format text-generation-session-execution-v2
     --text-file /raid/kanzhu/ServingStudio/data/corpus/enwik9 --tokenizer "$CKPT/tokenizer.json"
     --model glm53-flash-pp4 --backend vllm-tokens --base-url "$URL" --temperature 0
     --arrival-mode trace-timed --stream-idle-timeout-secs 600 --stop-session-on-error true --timeline true
     --log-path "$RD/requests.jsonl" --summary-path "$RD/summary.json" --timeline-path "$RD/timeline.parquet"
     --max-concurrency "$N" --max-model-len 1048576)
echo "${CMD[*]}" > "$RD/client_command.txt"
sha256sum "$CLIENT" >> "$RD/client_command.txt"
C0=$(date +%s)
"${CMD[@]}" > "$RD/client.log" 2>&1 &
CL=$!
while kill -0 $CL 2>/dev/null; do            # /metrics every 60 s (queue, preemption, prefix and offload counters)
  e=$(( $(date +%s) - C0 )); curl -s -m 10 "http://127.0.0.1:$P/metrics" > "$RD/prom/t$(printf %04d $e).prom"
  [ "$POLICY" != none ] && curl -s -m 10 "http://127.0.0.1:$((P + 30))/metrics" > "$RD/prom/router_t$(printf %04d $e).prom"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed "s/^/$e,/" >> "$RD/gpu_samples.csv"
  for _ in $(seq 60); do kill -0 $CL 2>/dev/null || break; sleep 1; done
done
wait $CL; rc=$?
echo "== client rc=$rc after $(( $(date +%s) - C0 )) s"
curl -s -m 10 "http://127.0.0.1:$P/metrics" > "$RD/prom/end.prom"
[ "$POLICY" != none ] && curl -s -m 10 "http://127.0.0.1:$((P + 30))/metrics" > "$RD/prom/router_end.prom"
kill $WD 2>/dev/null
$PY "$H/metrics.py" "$RD/requests.jsonl" > "$RD/metrics.json" && cat "$RD/metrics.json"
$PY -m bench.vllm_dpep_summary "$RD" --trace "$TRACE" || true
[ -f "$RD/INVALID.txt" ] && echo "== run invalid: $(cat "$RD/INVALID.txt")"
uptime
echo "== done $(date -Is)"
[ -f "$RD/INVALID.txt" ] && exit 4
(( rc == 0 || rc == 124 )) || exit $rc        # 124: the client stopped by its timeout, the normal end
exit 0
}
