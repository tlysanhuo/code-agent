#!/usr/bin/env bash
# GPU incursion watcher (incident #2, 2026-09-26: a foreign 63GB process
# landed on our colocate card during the rollout phase -- memory released by
# torch_memory_saver makes the card look free -- and OOM'd the first train
# forward). Alerts when a compute-app pid that was not present at baseline
# appears on one of our cards. Baseline is captured --after N minutes so the
# trainer's own processes are all up. No kills: alert only, humans decide.
#
# Usage: setsid nohup bash scripts/watch_gpu_incursion.sh 3,4,6,7 10 \
#          >> logs/rl-round1-gpu-watch.log 2>&1 &
set -u
IFS=',' read -ra IDX <<< "${1:?usage: watch_gpu_incursion.sh <gpu-indices> <baseline-min>}"
BASELINE_MIN="${2:-10}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runtime/agent-rl/round1-gpu-watch.log"

uuids() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v idx="$*" 'BEGIN{split(idx,a," ");for(i in a)want[a[i]]=1} want[$1]{print $2}'
}
pids_on() {  # pids of compute-apps on the given uuid set
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | awk -F', ' -v u="$1" 'BEGIN{n=split(u,L," ");for(i=1;i<=n;i++)want[L[i]]=1} want[$1]{print $2}' \
    | sort -u
}

U="$(uuids "${IDX[*]}")"
echo "[$(date -u '+%F %T')] watching uuids: $(echo $U | tr '\n' ' ')" | tee -a "$OUT"
echo "[$(date -u '+%F %T')] baseline capture in ${BASELINE_MIN}min" | tee -a "$OUT"
sleep $((BASELINE_MIN * 60))
BASE="$(pids_on "$U" | tr '\n' ' ')"
echo "[$(date -u '+%F %T')] baseline pids: $BASE" | tee -a "$OUT"

while true; do
  CUR="$(pids_on "$U")"
  NEW="$(comm -13 <(echo "$BASE" | tr ' ' '\n' | sort -u) <(echo "$CUR" | tr ' ' '\n' | sort -u) | grep -v '^$')"
  if [ -n "$NEW" ]; then
    echo "[$(date -u '+%F %T')] *** INCURSION: new pid(s) $NEW on our cards (baseline: $BASE)" | tee -a "$OUT"
  fi
  # absorb seen pids into baseline (alert once per new pid); the assignment
  # must NOT sit in a pipeline segment -- it would run in a subshell and the
  # parent's BASE would never update (the 1771-false-alert bug, fixed 09-27)
  BASE="$( (echo "$BASE" | tr ' ' '\n' | grep -v '^$'; echo "$CUR") | tr '\n' ' ')"
  BASE="$(echo $BASE | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')"
  sleep 30
done
