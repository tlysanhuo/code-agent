#!/usr/bin/env bash
# Stage-2 DSH agentic-GRPO smoke config (print-only by default; --launch is
# REFUSED until the user approves hyperparameters + GPU allocation).
# Scale-down of vendor/slime/examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
# (8x8 cards, 35B-A3B) to 2xH100 colocated for the dense 9B SFT checkpoint.
# Rollout: DSH via slime_dsh.generate (harness=7-passed, local sandbox backend,
# frozen gym evaluator as reward - gold=1.0/empty=0.0 integration-tested).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-0,5}"
SFT_MCORE="$PWD/models/dense-9B-sft/formal-2card/iter_0000499"
STUDENT_HF="$PWD/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PROMPT_DATA="$PWD/data/agent-rl/rl-round1-prompts.parquet"
SAVE_DIR="$PWD/models/dense-9B-rl/round1"
SLIME="$PWD/vendor/slime"

# ---- proposed hyperparameters (official recipe values unless noted) ----
LR=1e-6                 # official
LR_DECAY=constant       # official
EPS_CLIP=0.2            # official
EPS_CLIP_HIGH=0.28      # official (DAPO clip-higher)
KL_LOSS_COEF=0.0        # official (no KL)
ADVANTAGE=grpo          # official
ROLLOUT_BATCH=16        # ours: 2 tasks x 8 group
N_SAMPLES=8             # GRPO group size (Klear/DeepSWE convention)
ROLLOUT_CTX=16384       # smoke: match SFT training context; raise to 32k later
AGENT_TIME_BUDGET=600   # ours: smoke budget (official 1800)
ROLLOUT_GUARD=900
EVAL_TIMEOUT=600
STEPS_PER_ROLLOUT=1     # official
MAX_ROLLOUT_ROUNDS="${MAX_ROLLOUT_ROUNDS:-2}"   # smoke only

export SWE_TRAIN_PROTOCOL=scaleswe
export SWE_EVAL_PROTOCOL=scaleswe
export AGENT_TIME_BUDGET_SEC="$AGENT_TIME_BUDGET"
export EVAL_TIMEOUT_SEC="$EVAL_TIMEOUT"
export ROLLOUT_GUARD_SEC="$ROLLOUT_GUARD"
export AGENT_BOOT_CONCURRENCY=2
export AGENT_BOOT_RETRIES=1

source "$SLIME/scripts/models/qwen3.5-9B.sh"

CMD=(
  .venv-train-rl/bin/python "$SLIME/train.py"
  --actor-num-nodes 1 --actor-num-gpus-per-node 2
  --rollout-num-gpus 2 --rollout-num-gpus-per-engine 2 --colocate
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$STUDENT_HF"
  --ref-load "$SFT_MCORE"
  --save "$SAVE_DIR" --save-interval 100
  --no-gradient-accumulation-fusion
  --custom-generate-function-path slime_dsh.generate.generate
  --prompt-data "$PROMPT_DATA" --input-key prompt
  --rollout-batch-size "$ROLLOUT_BATCH" --n "$N_SAMPLES"
  --rollout-max-context-len "$ROLLOUT_CTX"
  --num-steps-per-rollout "$STEPS_PER_ROLLOUT"
  --num-epoch "$MAX_ROLLOUT_ROUNDS"
  --advantage-estimator "$ADVANTAGE" --kl-loss-coef "$KL_LOSS_COEF" --kl-coef 0.0
  --eps-clip "$EPS_CLIP" --eps-clip-high "$EPS_CLIP_HIGH"
  --lr "$LR" --lr-decay-style "$LR_DECAY" --min-lr "$LR"
  --optimizer adam --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95
  --optimizer-cpu-offload --use-precision-aware-optimizer
  --tensor-model-parallel-size 2 --sequence-parallel
  --pipeline-model-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu 16384
  --use-wandb --wandb-project code-agent-dense-mainline
  --wandb-group dense-9b-agent-rl-smoke
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash
)

echo "==== stage-2 GRPO smoke (PRINT-ONLY; not launched) ===="
printf '%s\n' "CUDA_VISIBLE_DEVICES=$GPUS"
printf ' %q' "${CMD[@]}"; echo
echo "env: SWE_* budgets above; rollout tasks from $PROMPT_DATA"
if [[ "${1:-}" == "--launch" ]]; then
  echo "REFUSED: hyperparameters pending user approval + GPU authorization." >&2
  echo "Approve via user review, then record allocation and remove this guard." >&2
  exit 2
fi
