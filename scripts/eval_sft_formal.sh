#!/usr/bin/env bash
# Post-stop eval runbook for the formal stage-1 SFT checkpoint.
# Prereq: runtime/sft-formal/stop-report.json with status STOPPED_AT_500 and
# conversion OK (produced by scripts/stop_sft_formal_at_500.sh).
#
# Phase A: serve models/dense-9B-sft/formal-2card/hf-iter500 with vLLM 0.19.1
#          using the EXACT P1 server params (port 18092, greedy eval, same
#          template) -> HumanEvalPlus first-32 base-check, directly comparable
#          to the 96.9% dense-9B baseline (runtime/loop-agent/p1-degradation.json).
# Phase B: offline vLLM NLL scoring of the prepared held-out splits
#          (runtime/sft-formal/nll-sft-formal-prepared.jsonl) for base 9B and
#          the SFT checkpoint.
#
# GPU discipline: single UUID argument; allocation rolled into
# configs/gpu-allocation.json; release verified at the end. The server is
# stopped before phase B and everything is torn down on exit.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

GPU_UUID="${1:?usage: eval_sft_formal.sh <gpu-uuid>}"
HF_CKPT="$PWD/models/dense-9B-sft/formal-2card/hf-iter500"
BASE_HF="$PWD/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
REPORT="$PWD/runtime/sft-formal/stop-report.json"
PREP="$PWD/runtime/sft-formal/nll-sft-formal-prepared.jsonl"
PY="$PWD/.venv-runtime/bin/python"
LOG="$PWD/logs/eval-sft-formal.log"
PORT=18092

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ---- preflight ----
"$PY" - "$REPORT" "$HF_CKPT" <<'PYEOF'
import json, sys, pathlib
rep = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert rep.get("status") == "STOPPED_AT_500", f"stop-report status: {rep.get('status')}"
hf = pathlib.Path(sys.argv[2])
assert (hf / "model.safetensors.index.json").is_file(), "converted HF dir missing"
print("preflight OK: checkpoint stopped at 500, HF conversion present")
PYEOF
[ -f "$PREP" ] || { echo "missing $PREP (run eval_sft_nll.py prepare first)" >&2; exit 1; }

# ---- GPU allocation roll ----
GPU_INDEX="$("$PY" - "$GPU_UUID" <<'PYEOF'
import json, sys, pathlib, datetime, subprocess
uuid = sys.argv[1]
p = pathlib.Path("configs/gpu-allocation.json")
cur = json.loads(p.read_text())
out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,index,memory.used", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout
rows = {r.split(",")[0].strip(): r.strip().split(",") for r in out.strip().splitlines()}
if uuid not in rows:
    sys.exit(f"UUID {uuid} not present: {list(rows)}")
used = int(rows[uuid][2])
if used > 512:
    sys.exit(f"GPU {uuid} is in use ({used} MiB); refusing")
rec = {
    "gpu_uuids": [uuid],
    "gpu_indices": [int(rows[uuid][1])],
    "allocated_by": "post-stop eval chain pinned by user (stop at step-500 ckpt -> convert -> HumanEvalPlus + held-out NLL); assistant-run runbook, idle verified at allocation",
    "allocated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "stage-1 SFT checkpoint eval: HumanEvalPlus first-32 base-check (P1 protocol) vs 96.9% baseline + held-out NLL (base vs SFT)",
    "previous_records": cur,
}
p.write_text(json.dumps(rec, indent=4) + "\n")
print(rows[uuid][1])
PYEOF
)"
log "allocation recorded: $GPU_UUID (index $GPU_INDEX)"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    log "stopping vLLM server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 3; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -f "vllm serve $HF_CKPT" 2>/dev/null || true
  sleep 5
  log "GPU state at teardown:"
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv | tee -a "$LOG"
}
trap cleanup EXIT

# ---- phase A: HumanEvalPlus (P1 protocol) ----
log "phase A: serving $HF_CKPT on :$PORT (P1 params)"
"$PWD/.venv-runtime/bin/vllm" serve "$HF_CKPT" \
  --tokenizer "$HF_CKPT" \
  --served-model-name DenseQwen-9B-SFT \
  --host 127.0.0.1 --port "$PORT" --api-key local-qwen \
  --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.8 --enforce-eager --language-model-only \
  --gdn-prefill-backend triton --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  >> "$PWD/logs/eval-sft-server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if curl -s -H "Authorization: Bearer local-qwen" "http://127.0.0.1:$PORT/v1/models" | grep -q DenseQwen-9B-SFT; then
    log "server up"; break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { log "FATAL: server died (see logs/eval-sft-server.log)"; exit 1; }
  sleep 10
done

"$PY" scripts/eval_loopify_humaneval.py --endpoint "http://127.0.0.1:$PORT/v1" \
  --model DenseQwen-9B-SFT --tag sft-formal-500 --limit 32 2>&1 | tee -a "$LOG"

kill "$SERVER_PID" 2>/dev/null || true
for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 3; done
kill -9 "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
log "phase A server stopped"

# ---- phase B: held-out NLL, base vs SFT ----
log "phase B: NLL scoring (base)"
"$PY" scripts/eval_sft_nll.py --score "$PREP" \
  --model-dir "$BASE_HF" --tag base --max-tokens 33024 2>&1 | tee -a "$LOG"
log "phase B: NLL scoring (sft-500)"
"$PY" scripts/eval_sft_nll.py --score "$PREP" \
  --model-dir "$HF_CKPT" --tag sft-500 --max-tokens 33024 2>&1 | tee -a "$LOG"

log "runbook complete; summaries in runtime/sft-formal/"
