#!/bin/bash
# run_call.sh TIMEOUT_S OUT ERR CMD...
# Run CMD in a new session (setsid), stdout to OUT and stderr to ERR; return its exit code, or 124
# on timeout. Cleanup touches only what this wrapper started:
#   (a) the process group the wrapper created (kill -- -PID), on timeout; and
#   (b) afterwards, every leftover pid whose session id (ps -o sid=) is that session.
# It never selects a process by GPU index or by user. If a pid of that session still holds a GPU
# after the kill, it logs this and returns 125 so the caller stops using the GPU; it kills
# nothing else.
t=$1 out=$2 err=$3; shift 3
setsid "$@" > "$out" 2> "$err" < /dev/null &
pid=$!
end=$(( $(date +%s) + t ))
rc=
while kill -0 "$pid" 2>/dev/null; do
  if (( $(date +%s) >= end )); then
    echo "run_call: timeout after ${t}s, killing process group $pid" >> "$err"
    kill -TERM -- "-$pid" 2>/dev/null; sleep 10; kill -KILL -- "-$pid" 2>/dev/null
    rc=124; break
  fi
  sleep 1
done
wait "$pid" 2>/dev/null; w=$?
[ -z "$rc" ] && rc=$w

session_pids() { ps -e -o pid=,sid= | awk -v s="$pid" '$2 == s && $1 != s {print $1}'; }
left=$(session_pids)
if [ -n "$left" ]; then
  echo "run_call: killing leftover pids of session $pid: $(echo $left)" >> "$err"
  kill -KILL $left 2>/dev/null
fi
# Wait up to 60 s until no compute process of this session is on any GPU (the driver can take a
# moment to release a killed process's context).
gpu_left=
for _ in $(seq 30); do
  gpu_left=
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' ')" = "$pid" ] && gpu_left="$gpu_left $p"
  done
  [ -z "$gpu_left" ] && break
  sleep 2
done
if [ -n "$gpu_left" ]; then
  echo "run_call: session $pid still holds a GPU (pids:$gpu_left) after cleanup; failing the call" >> "$err"
  exit 125
fi
exit $rc
