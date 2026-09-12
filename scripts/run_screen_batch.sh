#!/usr/bin/env bash
# Difficulty screening batch orchestrator (user-approved 2026-09-12: full batch, 1x idle H100).
# Waits for the qualification run to finish, serves Qwen3.5-27B, runs screen_rl_tasks.py,
# then stops the server and verifies GPU release. Detached-friendly: logs to logs/screen-batch.log.
set -euo pipefail
cd -- "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

QUAL_PID="${1:-4078403}"
PORT=18095
BASE_URL="http://127.0.0.1:${PORT}/v1"
GPU_INDICES_ALLOWED=(2 5)   # user-approved candidates only

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" ; }

say "watcher up; waiting for qualification PID ${QUAL_PID}"
while kill -0 "${QUAL_PID}" 2>/dev/null; do sleep 60; done
say "qualification process exited"

# final registry write (>=50 qualified tasks expected; wait up to 10 min)
for _ in $(seq 1 60); do
  if python3 - <<'EOF'
import json, sys
try:
    r = json.load(open('configs/agent-rl/local-task-registry.json'))
    sys.exit(0 if len(r) >= 50 else 1)
except Exception:
    sys.exit(1)
EOF
  then break; fi
  sleep 10
done
QUAL_N=$(python3 -c "import json;print(len(json.load(open('configs/agent-rl/local-task-registry.json'))))")
say "registry ready: ${QUAL_N} qualified tasks"

GPU=""
for idx in "${GPU_INDICES_ALLOWED[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${idx}")
  if [ "${used}" -lt 1000 ]; then GPU="${idx}"; break; fi
done
if [ -z "${GPU}" ]; then say "ABORT: no idle GPU among ${GPU_INDICES_ALLOWED[*]}"; exit 1; fi
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU}")
say "claiming GPU ${GPU} (${UUID})"

python3 - <<EOF
import json, time
alloc = {
  "gpu_uuids": ["${UUID}"], "gpu_indices": [${GPU}],
  "allocated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "allocated_by": "User approved 2026-09-12: difficulty screening full batch, 1x idle H100 (GPU 2 or 5)",
  "purpose": "27B teacher difficulty screening rollouts (DSH, k=4, LEGO-RL band)",
  "service": "vLLM Qwen3.5-27B BF16 port ${PORT} + screen_rl_tasks.py",
}
open('configs/screen-gpu-allocation.json','w').write(json.dumps(alloc, indent=1)+'\n')
EOF

export CUDA_VISIBLE_DEVICES="${GPU}"
bash scripts/start_qwen.sh --config configs/screen-qwen-server.json \
     --allocation configs/screen-gpu-allocation.json \
     > logs/screen-qwen-server.log 2>&1 &
SERVER_PID=$!
say "server launching pid=${SERVER_PID} (log: logs/screen-qwen-server.log)"

for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then say "ABORT: server exited early"; tail -30 logs/screen-qwen-server.log; exit 1; fi
  if curl -sf -H "Authorization: Bearer local-qwen" "${BASE_URL%/v1}/models" >/dev/null 2>&1; then break; fi
  sleep 10
done
say "server ready at ${BASE_URL}"

say "starting screening batch (this also validates the pipeline on its first attempt)"
.venv-train-rl/bin/python scripts/screen_rl_tasks.py --base-url "${BASE_URL}" \
    --k 4 --concurrency 3 > logs/screen-rl-tasks.log 2>&1
SCREEN_RC=$?
say "screening finished rc=${SCREEN_RC}"

say "stopping server pgid"
PGID=$(ps -o pgid= -p "${SERVER_PID}" | tr -d ' ')
kill -TERM -"${PGID}" 2>/dev/null || true
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
  [ "${used}" -lt 1000 ] && break
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
if [ "${used}" -ge 1000 ]; then
  say "TERM insufficient; escalating to KILL on our pgid ${PGID}"
  kill -KILL -"${PGID}" 2>/dev/null || true
  sleep 10
fi
final=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")
say "release check GPU ${GPU}: ${final} MiB"
say "batch orchestrator done"
