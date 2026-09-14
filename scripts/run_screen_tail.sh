#!/usr/bin/env bash
# Screening tail: run the controller against the already-running 27B server,
# then stop the server (pgid of vllm pid file owner) and verify GPU release.
set -uo pipefail
cd -- "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" ; }
GPU=2
.venv-train-rl/bin/python scripts/screen_rl_tasks.py --base-url "http://127.0.0.1:18095/v1" \
    --k 4 --concurrency 3 > logs/screen-rl-tasks.log 2>&1
say "screening controller rc=$?"
SERVER_PID=$(pgrep -f "vllm serve.*18095" | head -1 || true)
if [ -n "${SERVER_PID}" ]; then
  PGID=$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ')
  say "stopping server pid=${SERVER_PID} pgid=${PGID}"
  kill -TERM -"${PGID}" 2>/dev/null || true
fi
for _ in $(seq 1 36); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
  [ "${used}" -lt 1000 ] && break
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
if [ "${used}" -ge 1000 ] && [ -n "${SERVER_PID:-}" ]; then
  kill -KILL -"$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ')" 2>/dev/null || true; sleep 10
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
fi
say "release check GPU ${GPU}: ${used} MiB"
