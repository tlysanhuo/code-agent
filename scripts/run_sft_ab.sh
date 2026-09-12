#!/usr/bin/env bash
# Stage-1 SFT value A/B: base vs SFT zero-shot SWE trajectories on the 20
# held-out validated tasks. Serves each model with IDENTICAL vLLM params and
# runs scripts/eval_sft_ab_trajectories.py against it; grading = frozen gym
# evaluator. One GPU, recorded; server stopped between arms and at the end.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

GPU_UUID="${1:?usage: run_sft_ab.sh <gpu-uuid>}"
BASE_HF="$PWD/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SFT_HF="$PWD/models/dense-9B-sft/formal-2card/hf-iter500"
PY="$PWD/.venv-runtime/bin/python"
VPY="$PWD/.venv-train-rl/bin/python"
PORT=18093
LOG="$PWD/logs/sft-ab-run.log"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

GPU_INDEX="$("$PY" - "$GPU_UUID" <<'PYEOF'
import json, sys, pathlib, datetime, subprocess
uuid = sys.argv[1]
p = pathlib.Path("configs/gpu-allocation.json")
cur = json.loads(p.read_text())
out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,index,memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout
rows = {r.split(",")[0].strip(): r.strip().split(",") for r in out.strip().splitlines()}
if uuid not in rows: sys.exit(f"UUID {uuid} not present")
if int(rows[uuid][2]) > 512: sys.exit(f"GPU {uuid} in use ({rows[uuid][2]} MiB)")
rec = {"gpu_uuids": [uuid], "gpu_indices": [int(rows[uuid][1])],
       "allocated_by": "SFT value A/B experiment (user ordered 2026-09-12): base vs SFT "
                       "zero-shot trajectories on 20 held-out tasks",
       "allocated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "purpose": "stage-1 SFT decisive eval: trajectory capability A/B",
       "previous_records": cur}
p.write_text(json.dumps(rec, indent=4) + "\n")
print(rows[uuid][1])
PYEOF
)"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
log "allocation recorded: $GPU_UUID (index $GPU_INDEX)"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 3; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  sleep 5
  log "GPU at teardown:"; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv | tee -a "$LOG"
}
trap cleanup EXIT

serve() {  # $1 = model dir
  log "serving $1 on :$PORT"
  "$PWD/.venv-runtime/bin/vllm" serve "$1" \
    --tokenizer "$1" --served-model-name ABModel \
    --host 127.0.0.1 --port "$PORT" --api-key local-qwen \
    --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend mp \
    --max-model-len 40960 --max-num-seqs 6 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 --enforce-eager --language-model-only \
    --gdn-prefill-backend triton \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >> "$PWD/logs/sft-ab-server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 90); do
    if curl -s -H "Authorization: Bearer local-qwen" "http://127.0.0.1:$PORT/v1/models" | grep -q ABModel; then
      log "server up"; return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { log "FATAL: server died"; exit 1; }
    sleep 10
  done
  log "FATAL: server never became ready"; exit 1
}

stop_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 3; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  log "server stopped"
}

log "=== arm 1: base ==="
serve "$BASE_HF"
"$VPY" scripts/eval_sft_ab_trajectories.py \
  --endpoint "http://127.0.0.1:$PORT/v1" --model-name ABModel \
  --model-dir "$BASE_HF" --tag base 2>&1 | tee -a "$LOG"
stop_server

log "=== arm 2: sft-500 ==="
serve "$SFT_HF"
"$VPY" scripts/eval_sft_ab_trajectories.py \
  --endpoint "http://127.0.0.1:$PORT/v1" --model-name ABModel \
  --model-dir "$SFT_HF" --tag sft-500 2>&1 | tee -a "$LOG"
stop_server

log "A/B complete; summaries: runtime/sft-ab/{base,sft-500}/summary.json"
