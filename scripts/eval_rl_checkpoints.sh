#!/usr/bin/env bash
# Mid-training offline eval for RL checkpoints (MiMo-style; lesson from round1).
# For each mcore iter under SAVE_DIR: convert -> HF (CPU), serve it with vLLM on
# ONE spare GPU, run scripts/eval_dsh_heldout.py (20 frozen held-out tasks, DSH
# path, frozen gym grading), stop the server, append a row to the report.
#
# Served model name is Qwen3.5-27B on purpose: it must equal the models.id that
# configs/gym-dsh.patch.yml declares (the proven screening pair). The label is
# cosmetic -- the served weights are whatever checkpoint is converted here.
#
# Usage:
#   bash scripts/eval_rl_checkpoints.sh <gpu-uuid> [--dry-run] [--iters 19,39]
#                                       [--k 1] [--hf-dir <dir> --tag sft500]
# --hf-dir skips conversion (eval an existing HF dir, e.g. the SFT baseline).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

GPU_UUID="${1:?usage: eval_rl_checkpoints.sh <gpu-uuid> [--dry-run] [--iters 19,39] [--k 1] [--hf-dir <dir> --tag sft500]}"
shift || true
DRY=0; ITERS=""; K=1; HF_DIR=""; TAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --iters) ITERS="${2:?}"; shift ;;
    --k) K="${2:?}"; shift ;;
    --hf-dir) HF_DIR="${2:?}"; shift ;;
    --tag) TAG="${2:?}"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

STUDENT_HF="$CODE_AGENT_ROOT/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SAVE_DIR="$CODE_AGENT_ROOT/models/dense-9B-rl/round1-gspo"
PORT=18097
REPORT="$CODE_AGENT_ROOT/reports/rl-checkpoint-evals.md"
LOG="$CODE_AGENT_ROOT/logs/rl-ckpt-eval-$(date -u '+%Y%m%dT%H%M%SZ').log"
SRV_PID=""

log() { echo "[$(date -u '+%H:%M:%S')] $*" | tee -a "$LOG"; }

serve() {  # $1 = HF dir; mirrors run_sft_ab.sh (A/B-proven vLLM params)
  log "serving $1 on :$PORT (gpu $GPU_UUID)"
  "$CODE_AGENT_ROOT/.venv-runtime/bin/vllm" serve "$1" \
    --tokenizer "$1" --served-model-name Qwen3.5-27B \
    --host 127.0.0.1 --port "$PORT" --api-key local-qwen \
    --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend mp \
    --max-model-len 40960 --max-num-seqs 6 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 --enforce-eager --language-model-only \
    --gdn-prefill-backend triton \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >> "$CODE_AGENT_ROOT/logs/rl-ckpt-eval-server.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 90); do
    if curl -s -H "Authorization: Bearer local-qwen" "http://127.0.0.1:$PORT/v1/models" \
        | grep -q Qwen3.5-27B; then log "server up"; return 0; fi
    kill -0 "$SRV_PID" 2>/dev/null || { log "FATAL: server died"; return 1; }
    sleep 10
  done
  log "FATAL: server never became ready"; return 1
}

stop_server() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 2; done
    kill -9 "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
  fi
  sleep 10
  log "GPU after teardown:"; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader \
    | tee -a "$LOG" | grep -F "$GPU_UUID" || true
}
trap stop_server EXIT

convert() {  # $1 iter dir, $2 HF out
  log "converting $1 -> $2 (CPU)"
  CUDA_VISIBLE_DEVICES= "$CODE_AGENT_ROOT/.venv-train-rl/bin/python" \
    "$CODE_AGENT_ROOT/vendor/slime/tools/convert_torch_dist_to_hf.py" \
    --input-dir "$1" --output-dir "$2" --origin-hf-dir "$STUDENT_HF" --force \
    >> "$LOG" 2>&1
}

eval_one() {  # $1 HF dir, $2 tag
  serve "$1"
  "$CODE_AGENT_ROOT/.venv-train-rl/bin/python" "$CODE_AGENT_ROOT/scripts/eval_dsh_heldout.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --tag "$2" --k "$K" --model-dir "$1" \
    2>&1 | tee -a "$LOG"
  stop_server
  local sum="$CODE_AGENT_ROOT/runtime/agent-rl/heldout-evals/$2/summary.json"
  local row
  row=$(python3 -c "
import json; s = json.load(open('$sum'))
print(f\"| {s['tag']} | {s['n_tasks']}x{s['k']} | {s['avg_at_k']:.1%} | {s['any_solved_tasks']}/{s['n_tasks']} | {s['finished_utc']} |\")" \
    2>>"$LOG" || echo "| $2 | (summary missing) | | | |")
  mkdir -p "$CODE_AGENT_ROOT/reports"
  touch "$REPORT"
  grep -q '^| tag ' "$REPORT" || \
    printf '| tag | tasks | avg@k | any-solved | utc |\n|---|---|---|---|---|\n' >> "$REPORT"
  echo "$row" >> "$REPORT"
  log "report row: $row"
}

if [ "$DRY" = 1 ]; then
  log "DRY RUN -- plan:"
  [ -n "$HF_DIR" ] && { log "  eval HF dir $HF_DIR tag ${TAG:-hf}"; exit 0; }
  if [ -n "$ITERS" ]; then
    TRG="$ITERS"
  else
    TRG=$(ls "$SAVE_DIR" 2>/dev/null | grep -oE '^iter_[0-9]+$' | grep -oE '[0-9]+' | sort -n | tr '\n' ',' | sed 's/,$//')
  fi
  [ -z "$TRG" ] && { log "  no checkpoints under $SAVE_DIR (fresh run not started yet)"; exit 0; }
  IFS=',' read -ra IT <<< "$TRG"
  for it in "${IT[@]}"; do
    it=$((10#$it))
    log "  iter $it: convert $SAVE_DIR/iter_$(printf '%07d' "$it") -> $SAVE_DIR/hf-iter$it; serve :$PORT; eval tag iter$it k=$K"
  done
  exit 0
fi

if [ -n "$HF_DIR" ]; then
  eval_one "$HF_DIR" "${TAG:-hf}"
  exit 0
fi

if [ -n "$ITERS" ]; then
  TRG="$ITERS"
else
  TRG=$(ls "$SAVE_DIR" 2>/dev/null | grep -oE '^iter_[0-9]+$' | grep -oE '[0-9]+' | sort -n | tr '\n' ',' | sed 's/,$//')
fi
[ -z "$TRG" ] && { log "no checkpoints under $SAVE_DIR"; exit 1; }
IFS=',' read -ra IT <<< "$TRG"
for it in "${IT[@]}"; do
  it=$((10#$it))  # strip leading zeros: printf %07d treats 0000019 as octal
  ITER_DIR="$SAVE_DIR/iter_$(printf '%07d' "$it")"
  HF_OUT="$SAVE_DIR/hf-iter$it"
  if [ -d "$HF_OUT" ] && [ -f "$HF_OUT/config.json" ]; then
    log "$HF_OUT already converted, reusing"
  else
    convert "$ITER_DIR" "$HF_OUT"
  fi
  eval_one "$HF_OUT" "iter$it"
done
log "all checkpoint evals complete -> $REPORT"
