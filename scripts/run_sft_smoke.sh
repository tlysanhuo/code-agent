#!/usr/bin/env bash
# Stage-1 SFT cold-start SMOKE (authorized 2026-09-11, GPUs 0+6, UUIDs in configs/gpu-allocation.json).
# Scope: 256-row klear-only 8k slice, 4 optimizer steps, 1 checkpoint save, wandb ONLINE.
# Hyperparameters = official slime retool SFT example defaults (+ optimizer CPU offload
# from the official SWE recipe for the 2-card budget). No hyperparameter may change
# without user consent.
#
# Ray hygiene on this SHARED machine (user directive 2026-09-11):
#   - dedicated GCS port + dashboard port, never the defaults;
#   - NEVER global `ray stop`/`pkill ray` (they kill other users' clusters);
#   - teardown kills only processes whose cmdline references OUR session temp dir.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
source scripts/env-train-gpu.sh
source configs/secrets/wandb.env

export CUDA_VISIBLE_DEVICES=0,6
export CUDA_HOME=/usr/local/cuda
export WANDB_MODE=online
export WANDB_DIR="$CODE_AGENT_ROOT/logs/wandb"
export NCCL_NVLS_ENABLE=0
export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,localhost"

RAY_PORT=20379
RAY_DASH_PORT=28266
RAY_BIN="$CODE_AGENT_ROOT/.venv-train-rl/bin/ray"

# ray session sockets must stay under the 107-byte AF_UNIX path limit; the
# volume tmp path is too long - use a short root-fs dir (small: logs+sockets only)
export RAY_TMPDIR=/tmp/ray-sft-smoke
mkdir -p "$RAY_TMPDIR"

scoped_ray_stop() {
  # kill ONLY this launcher's ray processes: match our session temp dir in cmdline,
  # plus ray:: workers that are descendants of our raylet. NEVER global ray stop.
  local pids=""
  pids+="$(ps aux | grep '[r]ay' | grep -F "$RAY_TMPDIR" | awk '{print $2}' | tr '\n' ' ' || true)"
  local raylet_pids child
  raylet_pids="$(ps aux | grep '[r]aylet' | grep -F "$RAY_TMPDIR" | awk '{print $2}' || true)"
  for rp in $raylet_pids; do
    for child in $(ps --ppid "$rp" -o pid --no-headers 2>/dev/null); do
      pids+="$child "
    done
  done
  pids=$(echo "$pids" | tr ' ' '\n' | sort -u | grep -v '^$' || true)
  if [ -n "$pids" ]; then
    echo "scoped stop: killing $pids" >&2
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 8
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  else
    echo "scoped stop: no ray processes under $RAY_TMPDIR" >&2
  fi
}

# data slice: first 256 rows of the klear-only 8k-bounded trainset -> 4 steps at gbs 64
SFT_DATA="$CODE_AGENT_ROOT/data/sft-pool/train-8k-klear-only.parquet@[0:256]"
STUDENT_HF="$CODE_AGENT_ROOT/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STUDENT_MCORE="$STUDENT_HF/torch_dist"
SAVE_DIR="$CODE_AGENT_ROOT/models/dense-9B-sft/smoke"
SLIME="$CODE_AGENT_ROOT/vendor/slime"

scoped_ray_stop
"$RAY_BIN" start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASH_PORT" \
  > "$CODE_AGENT_ROOT/logs/sft-smoke-ray.log" 2>&1
sleep 5

source "$SLIME/scripts/models/qwen3.5-9B.sh"

RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
env = {
  "PYTHONPATH": f"{os.environ.get('PYTHONPATH','')}",
  "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
  "NCCL_NVLS_ENABLE": "0",
  "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
  "WANDB_MODE": "online",
  "WANDB_DIR": os.environ["WANDB_DIR"],
  "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
  "HF_HOME": os.environ.get("HF_HOME", ""),
  "MASTER_ADDR": "127.0.0.1",
}
print(json.dumps({"env_vars": {k: v for k, v in env.items() if v}}))
PY
)

set +e
"$RAY_BIN" job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- "$CODE_AGENT_ROOT/.venv-train-rl/bin/python" "$SLIME/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "$STUDENT_HF" \
   --ref-load "$STUDENT_MCORE" \
   --save "$SAVE_DIR" \
   --save-interval 4 \
   --no-gradient-accumulation-fusion \
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
   --prompt-data "$SFT_DATA" \
   --input-key messages \
   --rollout-shuffle \
   --num-epoch 1 \
   --rollout-batch-size 64 \
   --global-batch-size 64 \
   --loss-type sft_loss \
   --loss-mask-type qwen3_5 \
   --calculate-per-token-loss \
   --disable-compute-advantages-and-returns \
   --debug-train-only \
   --tensor-model-parallel-size 1 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 8192 \
   --optimizer adam \
   --lr 1e-5 \
   --lr-decay-style cosine \
   --min-lr 1e-6 \
   --lr-warmup-fraction 0.1 \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.95 \
   --optimizer-cpu-offload \
   --overlap-cpu-optimizer-d2h-h2d \
   --use-precision-aware-optimizer \
   --use-wandb \
   --wandb-project code-agent-dense-mainline \
   --wandb-group dense-9b-sft-smoke \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   2>&1 | tee "$CODE_AGENT_ROOT/logs/sft-smoke-train.log"
rc=${PIPESTATUS[0]}
set -e

scoped_ray_stop
exit $rc
