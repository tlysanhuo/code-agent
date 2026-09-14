#!/usr/bin/env bash
# round1b full screening tail (user-approved 2026-09-14, GPU 0, concurrency 6):
# runs the controller against the already-running 27B server (port 18095), then
# stops the server and verifies GPU 0 release. Detached-friendly.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" ; }
GPU=0
ROOT=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent

OUT_DIR="$ROOT/runtime/agent-rl/round1b-screen"
mkdir -p "$OUT_DIR"
# carry over the 24 trial attempts (same protocol/tasks) so they are skipped
if [ -f "$ROOT/runtime/agent-rl/round1b-screen-trial/results.jsonl" ] && ! [ -f "$OUT_DIR/results.jsonl" ]; then
  cp "$ROOT/runtime/agent-rl/round1b-screen-trial/results.jsonl" "$OUT_DIR/results.jsonl"
  say "seeded results.jsonl with 24 trial attempts"
fi

.venv-train-rl/bin/python scripts/screen_rl_tasks.py --base-url "http://127.0.0.1:18095/v1" \
    --registry "$ROOT/configs/agent-rl/local-task-registry-r1b-merged.json" \
    --prompts "$ROOT/data/agent-rl/rl-round1b-merged-prompts.parquet" \
    --out-dir "$OUT_DIR" \
    --screened-json "$ROOT/configs/agent-rl/rl-round1b-screened.json" \
    --screened-parquet "$ROOT/data/agent-rl/rl-round1b-screened-prompts.parquet" \
    --k 4 --concurrency 6 > "$ROOT/logs/screen-r1b-full.log" 2>&1
say "screening controller rc=$?"

SERVER_PID=$(pgrep -f "vllm serve.*18095|launch_qwen.*r1b" | head -1 || true)
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
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader -i "${GPU}" > "$ROOT/runtime/agent-rl/round1b-screen/gpu-release.txt" 2>/dev/null || true
say "round1b full screening tail done"
