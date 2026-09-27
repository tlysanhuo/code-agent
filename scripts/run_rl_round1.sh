#!/usr/bin/env bash
# Stage-2 ROUND-1 GSPO launch — faithful 4xH100 scale-down of the official
# vendor/slime/examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
# (user-approved 2026-09-16: "去改" after the deviation-table review).
# Canonical bits (quick_start.md CKPT_ARGS / ROLLOUT_ARGS, example SGLANG_ARGS,
# FAQ#3/#8/#10, agent.md adapter parsers) — every flag matches upstream docs.
# Ray hygiene: dedicated ports + short tmpdir (AF_UNIX 107B limit), scoped teardown.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
source scripts/env-train-gpu.sh
set -a; source configs/secrets/wandb.env; set +a

export CUDA_VISIBLE_DEVICES=3,4,6,7
export WANDB_MODE=online
export WANDB_DIR="$CODE_AGENT_ROOT/logs/wandb"
export NCCL_NVLS_ENABLE=0
export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,localhost"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# anti-fragmentation (incidents #2/#4: train-step 14.3GiB spike lost to a
# reserved-but-unallocated 20GiB hole plus an 8.65GiB foreign squatter;
# torch's own recommendation from the OOM traceback)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# dp_schedule alignment: one 19k-token sample per mb -> mbs == samples; any
# 2-segment session makes the count odd and DP=2 alignment asserts (67->68).
# Raising the REALIGN absorption window keeps sessions single-sample (merge
# stays dormant: non-thinking echoes match, no dict rewrites to absorb).
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=8192
export PYTHONPATH="$CODE_AGENT_ROOT:$CODE_AGENT_ROOT/vendor/slime${PYTHONPATH:+:$PYTHONPATH}"

# ---- DSH rollout knobs (upstream example env names; budgets per approved sheet) ----
# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (official example:
# not 127.0.0.1) — use this node's primary IP.
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-$(hostname -I | awk '{print $1}')}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
export SWE_AGENT_TIME_BUDGET_SEC=1800
export SWE_EVAL_TIMEOUT_SEC=600
export SWE_BOOT_CONCURRENCY=8
export SWE_BOOT_RETRIES=2
# ---- reward stack v1 (approved: binary core, shaping OFF, blocker ON) ----
export DSH_BLOCKER=1
export DSH_C_UNFINISHED=0
export DSH_C_FMT=0
export DSH_GROUP_REPAIR=0
# rewrite-merge DESTROYS short turns' TurnRecords (turn=None) -- every turn
# but the last became untrainable. 0 disables the merge (see
# _try_merge_assistant_rewrite "feature off"); rewrites then fork, and the
# every turn natively (aligned with SFT/A-B/screening, all non-thinking)

# canonical CKPT_ARGS: hf-checkpoint for tokenizer only; ref-load = SFT mcore
# (fresh start: --load unset -> slime falls back to ref-load with fresh optimizer)
SFT_HF="$CODE_AGENT_ROOT/models/dense-9B-sft/formal-2card/hf-iter500"
REF_MODEL_PATH="$CODE_AGENT_ROOT/models/dense-9B-sft/formal-2card"
PROMPT_DATA="$CODE_AGENT_ROOT/data/agent-rl/rl-round1-band-prompts.parquet"
SAVE_DIR="$CODE_AGENT_ROOT/models/dense-9B-rl/round1-gspo"
SLIME="$CODE_AGENT_ROOT/vendor/slime"
mkdir -p "$SAVE_DIR/rollout_dumps"

# auto-resume: when a checkpoint exists, continue from it instead of
# fresh-starting from ref-load (slime restores weights + optimizer + data offset
# from latest_checkpointed_iteration.txt)
LOAD_ARGS=()
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  LOAD_ARGS=(--load "$SAVE_DIR")
  echo "[$(date -u '+%H:%M:%S')] RESUME from iter $(cat "$SAVE_DIR/latest_checkpointed_iteration.txt") in $SAVE_DIR" >&2
fi

RAY_PORT=24379
RAY_DASH_PORT=32266
RAY_BIN="$CODE_AGENT_ROOT/.venv-train-rl/bin/ray"
# Per-generation tmpdir (incident #12: two launch generations shared one
# RAY_TMPDIR and each one's cleanup killed the other's live ray head). The pid
# suffix makes every cleanup provably own-scoped and lets later launches sweep
# orphaned ray heads of DEAD generations. Path stays well under the 107B
# AF_UNIX limit even with the suffix + ray's session subdir.
export RAY_TMPDIR="/tmp/ray-rl-r1-g$$"
rm -rf "$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"

# kill ray processes under dir $1 (+ raylet children). Matching by our unique
# path prefix never touches other users' ray clusters on this shared box.
stop_ray_under() {
  local dir="$1" pids="" raylet_pids rp child
  pids+="$(ps aux | grep '[r]ay' | grep -F "$dir" | awk '{print $2}' | tr '\n' ' ' || true)"
  raylet_pids="$(ps aux | grep '[r]aylet' | grep -F "$dir" | awk '{print $2}' || true)"
  for rp in $raylet_pids; do
    for child in $(ps --ppid "$rp" -o pid --no-headers 2>/dev/null); do
      pids+="$child "
    done
  done
  pids=$(echo "$pids" | tr ' ' '\n' | sort -u | grep -v '^$' || true)
  if [ -n "$pids" ]; then
    echo "[$(date -u '+%H:%M:%S')] scoped stop ($dir): killing $pids" >&2
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true; sleep 8
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

scoped_ray_stop() { stop_ray_under "$RAY_TMPDIR"; }

# An exec'd trainer has no bash EXIT trap, so a dead generation leaves its ray
# head daemons behind holding port 24379. Sweep them at startup — but ONLY
# dirs whose generation pid is provably dead: a live generation's head is
# never touched (the exact #12 failure mode), other users' ray never matches.
sweep_dead_generations() {
  local d pid
  for d in /tmp/ray-rl-r1-g[0-9]*; do
    [ -d "$d" ] || continue
    pid="${d##*-g}"
    kill -0 "$pid" 2>/dev/null && continue
    echo "[$(date -u '+%H:%M:%S')] sweeping dead generation $d" >&2
    stop_ray_under "$d"
    rm -rf "$d"
  done
  # legacy fixed path from pre-#12 scripts. Guarded by "no trainer lineage
  # alive" — under a (forbidden) concurrent manual launch this cleans nothing
  # and our own ray start then fails loudly on the busy port instead of
  # killing the live generation.
  if [ -d /tmp/ray-rl-r1 ] && ! pgrep -f "vendor/slime/train[.]py" >/dev/null 2>&1; then
    echo "[$(date -u '+%H:%M:%S')] sweeping legacy /tmp/ray-rl-r1 (no live lineage)" >&2
    stop_ray_under /tmp/ray-rl-r1
    rm -rf /tmp/ray-rl-r1
  fi
}
trap 'scoped_ray_stop; rm -rf "$RAY_TMPDIR"; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader -i 3,4,6,7 | tee "$CODE_AGENT_ROOT/runtime/agent-rl/round1-gspo-gpu-release.txt" || true' EXIT

sweep_dead_generations
scoped_ray_stop
"$RAY_BIN" start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASH_PORT" \
  > "$CODE_AGENT_ROOT/logs/rl-round1-ray.log" 2>&1
sleep 5

source "$SLIME/scripts/models/qwen3.5-9B.sh"

CMD=(
  .venv-train-rl/bin/python "$SLIME/train.py"
  --actor-num-nodes 1 --actor-num-gpus-per-node 4
  --rollout-num-gpus 4 --rollout-num-gpus-per-engine 4 --colocate
  "${MODEL_ARGS[@]}"

  --hf-checkpoint "$SFT_HF"
  --ref-load "$REF_MODEL_PATH"
  --save "$SAVE_DIR" --save-interval 10
  "${LOAD_ARGS[@]}"

  --custom-generate-function-path slime_dsh.generate.generate
  # DP-schedule parity guard (incident #8): forked duplicate renderings can
  # leave an ODD total sample count, which build_dp_schedule cannot align to
  # dp_size and crashes the run; the filter drops redundant fork siblings only
  # when needed (even totals = strict no-op). Verified on the crashing batch.
  --rollout-sample-filter-path slime_dsh.dp_parity.sample_filter
  # advantage processing: UPSTREAM DEFAULT (official example behavior; custom
  # group-norm hook removed after incident #7 — align with the tested recipe)
  --no-gradient-accumulation-fusion

  --prompt-data "$PROMPT_DATA" --input-key prompt --metadata-key metadata
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --num-rollout 100
  --rollout-batch-size 8 --n-samples-per-prompt 8
  --num-steps-per-rollout 1 --global-batch-size 64
  --rollout-max-context-len 32768 --rollout-max-response-len 8192
  --rollout-temperature 1.0
  --rollout-stop-token-ids 248046 248044
  --rollout-shuffle --balance-data
  # DAPO dynamic sampling DROPPED for v1: pinned slime's filter_hub assumes flat
  # sample lists while agent fan-out returns nested sibling lists
  # (sglang_rollout.py:460); official SWE example also runs without it.
  --save-debug-rollout-data "$SAVE_DIR/rollout_dumps/rollout_{rollout_id}.pt"
  --micro-batch-size 1

  --advantage-estimator gspo --kl-loss-coef 0.0 --kl-loss-type low_var_kl --kl-coef 0.0 --entropy-coef 0.0
  --eps-clip 0.2 --eps-clip-high 0.28

  --optimizer adam --lr 1e-6 --lr-decay-style constant --min-lr 1e-6
  --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer

  --tensor-model-parallel-size 2 --sequence-parallel
  --pipeline-model-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 8192
  --log-probs-chunk-size 1024

  --sglang-mem-fraction-static 0.8
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3

  --use-wandb --wandb-project code-agent-dense-mainline
  --wandb-group stage2-gspo-round1
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash
)

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] stage-2 round1 GSPO (official-recipe port): 4xH100 GPUs 3,4,6,7, band=103" >&2
printf ' %q' "${CMD[@]}"; echo
exec "${CMD[@]}"
