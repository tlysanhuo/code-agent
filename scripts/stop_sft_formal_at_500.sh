#!/usr/bin/env bash
# Watcher: stop formal stage-1 SFT right after the step-500 checkpoint (= iteration 499, 0-indexed saves) lands
# (user-pinned 1 epoch; job itself is configured for 882 steps, so an external
# stop is required). Detached via setsid by the operator; safe to re-run
# (flock). SHARED MACHINE: teardown is scoped to RAY_TMPDIR=/tmp/ray-sft-smoke
# session-dir matches ONLY. Never a global ray stop / pkill.
#
# Sequence:
#   1. poll until iter_0000499 is complete (.metadata + latest file >= 499
#      + size stable across two polls)
#   2. scoped teardown (SIGTERM -> 8s -> SIGKILL), same logic as
#      run_sft_formal.sh:scoped_ray_stop
#   3. verify no session processes remain + GPU 0/5 release evidence
#   4. convert mcore -> HF on CPU (validated command, smoke-proven)
#   5. write runtime/sft-formal/stop-report.json
# Does NOT start any GPU eval/serving; that stays with the operator.
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RAY_TMPDIR=/tmp/ray-sft-smoke
CKPT_ROOT="$PWD/models/dense-9B-sft/formal-2card"
TARGET_ITER=499
LOG="$PWD/logs/sft-formal-stop.log"
REPORT_DIR="$PWD/runtime/sft-formal"
LOCK="$PWD/logs/sft-formal-stop.lock"
RAY_BIN="$PWD/.venv-train-rl/bin/ray"
PY="$PWD/.venv-train-rl/bin/python"
STUDENT_HF="$PWD/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another watcher instance holds $LOCK; exiting" >&2
  exit 3
fi

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

iter_saved() {
  local latest
  latest="$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
  [ "${latest:-0}" -ge "$TARGET_ITER" ] 2>/dev/null || return 1
  [ -f "$CKPT_ROOT/iter_0000499/.metadata" ] || return 1
  return 0
}

session_pids() {
  # exact scoped match from run_sft_formal.sh, plus transitive children so the
  # job driver (train.py, whose cmdline lacks the session dir) is included;
  # exclude our own pid tree
  local me="$$ $PPID"
  local all new children p round
  all="$({ ps aux | grep '[r]ay' | grep -F "$RAY_TMPDIR" | awk '{print $2}'
    for rp in $(ps aux | grep '[r]aylet' | grep -F "$RAY_TMPDIR" | awk '{print $2}'); do
      ps --ppid "$rp" -o pid --no-headers 2>/dev/null
    done
  } 2>/dev/null | sort -un | tr '\n' ' ')"
  for round in 1 2 3 4; do
    children=""
    for p in $all; do
      children="$children $(ps --ppid "$p" -o pid --no-headers 2>/dev/null | tr '\n' ' ')"
    done
    new="$(printf '%s %s\n' "$all" "$children" | tr ' ' '\n' | sort -un | grep -v '^$' | tr '\n' ' ')"
    [ "$new" = "$all" ] && break
    all="$new"
  done
  for p in $all; do
    case " $me " in *" $p "*) ;; *) echo "$p";; esac
  done
}

# ---- phase 1: wait for the checkpoint ----
log "watcher started (pid $$), waiting for $CKPT_ROOT/iter_0000499"
while true; do
  if iter_saved; then
    s1="$(du -sb "$CKPT_ROOT/iter_0000499" 2>/dev/null | cut -f1)"
    sleep 90
    s2="$(du -sb "$CKPT_ROOT/iter_0000499" 2>/dev/null | cut -f1)"
    if [ -n "$s1" ] && [ "$s1" = "$s2" ]; then
      log "checkpoint 500 complete and stable (${s2} bytes)"
      break
    fi
    log "checkpoint still being written ($s1 -> $s2), keep waiting"
  elif [ -z "$(session_pids)" ]; then
    log "FATAL: training session processes are gone before iter_0000499 landed"
    tail -5 "$PWD/logs/sft-formal-train.log" >> "$LOG" 2>&1
    mkdir -p "$REPORT_DIR"
    printf '{"status":"FAILED_EARLY_TRAINING_LOST","utc":"%s"}\n' \
      "$(date -u +%FT%TZ)" > "$REPORT_DIR/stop-report.json"
    exit 1
  fi
  sleep 120
done

# ---- phase 2: scoped teardown ----
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv >> "$LOG" 2>&1
pids="$(session_pids | tr '\n' ' ')"
if [ -n "${pids// /}" ]; then
  log "scoped stop: SIGTERM to $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 8
  pids2="$(session_pids | tr '\n' ' ')"
  if [ -n "${pids2// /}" ]; then
    log "scoped stop: SIGKILL to $pids2"
    # shellcheck disable=SC2086
    kill -9 $pids2 2>/dev/null || true
  fi
else
  log "scoped stop: no processes under $RAY_TMPDIR (job already finished?)"
fi

# run_sft_formal.sh runs its own scoped_ray_stop when the submit client
# returns; give it time, then require silence.
for _ in $(seq 1 15); do
  [ -z "$(session_pids)" ] && break
  sleep 10
done
if [ -n "$(session_pids)" ]; then
  log "WARN: session processes still alive after teardown:"
  session_pids | while read -r p; do ps -o pid,args -p "$p" >> "$LOG" 2>&1; done
fi

# ---- phase 3: GPU release evidence ----
sleep 20
log "GPU state after teardown:"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv >> "$LOG" 2>&1
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv >> "$LOG" 2>&1

# ---- phase 4: mcore -> HF conversion, CPU-forced ----
HF_OUT="$CKPT_ROOT/hf-iter500"
log "converting $CKPT_ROOT/iter_0000499 -> $HF_OUT (CPU)"
if CUDA_VISIBLE_DEVICES= "$PY" vendor/slime/tools/convert_torch_dist_to_hf.py \
     --input-dir "$CKPT_ROOT/iter_0000499" \
     --output-dir "$HF_OUT" \
     --origin-hf-dir "$STUDENT_HF" \
     --force >> "$PWD/logs/convert-formal-500-to-hf.log" 2>&1; then
  log "conversion OK -> $HF_OUT"
  conv_status=OK
else
  log "conversion FAILED (see logs/convert-formal-500-to-hf.log)"
  conv_status=FAILED
fi

# ---- phase 5: report ----
mkdir -p "$REPORT_DIR"
"$PY" - "$REPORT_DIR/stop-report.json" <<'PYEOF' >> "$LOG" 2>&1
import json, subprocess, sys, datetime, pathlib
root = pathlib.Path.cwd()
out = {
    "status": "STOPPED_AT_500",
    "utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "checkpoint": str(root / "models/dense-9B-sft/formal-2card/iter_0000499"),
    "hf_dir": str(root / "models/dense-9B-sft/formal-2card/hf-iter500"),
    "conversion": None,
    "gpu_after": subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines(),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
PYEOF
log "watcher done (conversion=$conv_status); eval/serving left to operator"
