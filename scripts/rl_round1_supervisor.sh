#!/usr/bin/env bash
# Round-1 detached watchdog (runlog #10/#11: external SIGTERM kept killing the
# trainer ~5h into each attempt; source not reachable from inside the box).
# Strategy: make kills harmless — save-interval 10 caps the loss at 10 steps,
# and this supervisor relaunches (auto-resume from the latest checkpoint)
# within 60s of any death. Fully detached from any shell session tree.
#
# Incident #12 hardening (2026-09-27):
#   - flock single-instance lock: two supervisors once double-launched the
#     trainer and the launch generations killed each other's ray head;
#   - lineage liveness = train.py OR run script: a fresh launch spends ~1min
#     in its prelude before train.py exists — that gap caused the #12 race;
#   - hang detection: a trainer wedged in ray.init's GCS retry loop is ALIVE
#     but deadlocked, and "alive" alone must not block recovery. The #12
#     signature (>= GCS_FAIL_KILL "Failed to connect to GCS" lines in the
#     current log; healthy runs log zero) is killed on sight so the normal
#     relaunch path takes over. Generic log staleness only alerts.
#
# Usage: setsid nohup bash scripts/rl_round1_supervisor.sh \
#          > /dev/null 2>&1 < /dev/null &
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

SUP_LOG=runtime/agent-rl/round1-supervisor.log
LOCK_FILE=/tmp/rl-round1-supervisor.lock
MAX_RESTARTS=5        # per window
WINDOW=3600           # seconds
HANG_STALE=1800       # log silent this long with lineage alive -> alert only
GCS_FAIL_KILL=6       # >= N GCS connect failures in current log -> certain hang
GPUS="${GPUS:-3,4,6,7}"
declare -a STAMPS=()
LAST_STALE_ALERT=0

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -u '+%F %T')Z] refusing to start: another supervisor holds $LOCK_FILE" >> "$SUP_LOG"
  exit 1
fi

gpus_free() {  # all our cards near-empty (no trainer AND no foreign squatter)
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null)
  [ -z "$mem" ] && return 1
  awk -v ok=1 '{ if ($1 > 2000) ok=0 } END { exit !ok }' <<< "$mem"
}

lineage_alive() {
  pgrep -f "vendor/slime/train[.]py" >/dev/null 2>&1 && return 0
  pgrep -f "run_rl_round1[.]sh"      >/dev/null 2>&1 && return 0
  return 1
}

log() { echo "[$(date -u '+%F %T')Z] $*" >> "$SUP_LOG"; }

log "supervisor up (pid $$), gpus=$GPUS, save-interval=10, budget=${MAX_RESTARTS}/${WINDOW}s, hang: stale>${HANG_STALE}s alert / gcs>=${GCS_FAIL_KILL} kill"

while true; do
  if lineage_alive; then
    # hang watch (incident #12): alive-but-wedged must also trigger recovery
    if [ -f logs/rl-round1-gspo.log ]; then
      gcs=$(grep -c "Failed to connect to GCS" logs/rl-round1-gspo.log 2>/dev/null || true)
      if [ "${gcs:-0}" -ge "$GCS_FAIL_KILL" ]; then
        log "HANG: lineage alive with $gcs GCS connect failures (incident-#12 signature) - clearing trainer for relaunch"
        pkill -f "vendor/slime/train[.]py" 2>/dev/null
        pkill -f "run_rl_round1[.]sh" 2>/dev/null
      else
        now=$(date +%s)
        mtime=$(stat -c %Y logs/rl-round1-gspo.log 2>/dev/null || echo "$now")
        if (( now - mtime > HANG_STALE )) && (( now - LAST_STALE_ALERT > 600 )); then
          log "ALERT: log silent $((now - mtime))s with lineage alive - possible hang, NOT auto-killing"
          LAST_STALE_ALERT=$now
        fi
      fi
    fi
  else
    now=$(date +%s)
    # prune restart stamps outside the window, then budget-check
    keep=()
    for t in ${STAMPS[@]+"${STAMPS[@]}"}; do
      [ -n "$t" ] && (( now - t < WINDOW )) && keep+=("$t")
    done
    STAMPS=(${keep[@]+"${keep[@]}"})
    if (( ${#STAMPS[@]} >= MAX_RESTARTS )); then
      log "HALT: ${#STAMPS[@]} restarts in the last ${WINDOW}s - something is systematically wrong, refusing to thrash. Manual check required."
      exit 1
    fi
    if gpus_free; then
      STAMPS+=("$now")
      log "trainer dead; GPUs free; relaunching (restart ${#STAMPS[@]}/$MAX_RESTARTS in window)"
      # sidecar follows the log by path; restart it so truncation is handled
      pkill -f "scripts/monitor_rl_run.py" 2>/dev/null
      setsid nohup bash scripts/run_rl_round1.sh > logs/rl-round1-gspo.log 2>&1 < /dev/null &
      sleep 5
      setsid nohup .venv-train-rl/bin/python scripts/monitor_rl_run.py \
        --log logs/rl-round1-gspo.log --out runtime/agent-rl/round1-monitor --follow \
        > /dev/null 2>&1 < /dev/null &
    else
      log "trainer dead but GPUs $GPUS NOT free (foreign load?) - waiting, no relaunch"
    fi
  fi
  sleep 60
done
