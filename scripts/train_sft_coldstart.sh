#!/usr/bin/env bash
# Stage 1 (SFT cold-start) draft for the dense mainline: Qwen3.5-9B student on
# agent trajectories (Klear-66k / OpenHands / R2E-Gym pools with exclusions).
# Default action PRINTS the planned commands; --launch is hard-gated on GPU
# authorization. Hyperparameters are copied from the official slime retool SFT
# example; any change requires user consent before a real run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

MODE="${1:---print}"

# --- fixed project paths -----------------------------------------------------
STUDENT_HF="models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STUDENT_MCORE="${STUDENT_HF}/torch_dist"        # produced by step 0 below (pending megatron-prep env)
SAVE_DIR="models/dense-9B-sft/slime"
SFT_DATA="data/sft-pool/train-16k.parquet"      # produced by scripts/make_sft_trainset.py --max-tokens 16384
SLIME_PIN='4c193f1f37509cca70f0e88807a9305b70f63f4e'

# --- guard: pinned slime, clean tree -----------------------------------------
check_pin() {
  [[ "$(git -C vendor/slime rev-parse HEAD)" == "$SLIME_PIN" ]] || { echo 'vendor/slime is not at the pinned commit' >&2; exit 2; }
  [[ -z "$(git -C vendor/slime status --porcelain)" ]] || { echo 'vendor/slime working tree is dirty' >&2; exit 2; }
}

# --- step 0: HF -> mcore conversion (once, needs megatron-prep environment) ---
print_convert_cmd() {
  cat <<EOF
# Prerequisite (megatron-prep env with pinned Megatron-LM 1dcf0da + slime megatron.patch):
cd vendor/slime
source scripts/models/qwen3.5-9B.sh
PYTHONPATH=<megatron-lm-src> python tools/convert_hf_to_torch_dist.py \\
    \${MODEL_ARGS[@]} \\
    --hf-checkpoint ../../${STUDENT_HF} \\
    --save ../../${STUDENT_MCORE}
EOF
}

# --- step 1: SFT training (official retool example structure, project paths) --
print_train_plan() {
  cat <<EOF
ray start --head ... # inside the training environment only, under GPU authorization

ray job submit --address="http://127.0.0.1:8265" \\
  --runtime-env-json='{"env_vars": {"PYTHONPATH": "<megatron-lm-src>/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \\
  -- python3 train.py \\
   --actor-num-nodes 1 \\
   --actor-num-gpus-per-node 3 \\
   --rollout-num-gpus 3 \\
   --colocate \\
   \${MODEL_ARGS[@]}                                   # sourced from vendor/slime/scripts/models/qwen3.5-9B.sh \\
   --hf-checkpoint ${STUDENT_HF} \\
   --ref-load ${STUDENT_MCORE} \\
   --save ${SAVE_DIR} \\
   --save-interval 100 \\
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout \\
   --prompt-data ${SFT_DATA} \\
   --input-key messages \\
   --rollout-shuffle \\
   --num-epoch 2 \\
   --rollout-batch-size 64 \\
   --global-batch-size 64 \\
   --loss-type sft_loss \\
   --loss-mask-type qwen3_5 \\
   --calculate-per-token-loss \\
   --disable-compute-advantages-and-returns \\
   --debug-train-only \\
   --tensor-model-parallel-size 1 \\
   --sequence-parallel \\
   --pipeline-model-parallel-size 1 \\
   --recompute-granularity full \\
   --recompute-method uniform \\
   --recompute-num-layers 1 \\
   --use-dynamic-batch-size \\
   --max-tokens-per-gpu 16384 \\
   --optimizer adam \\
   --lr 1e-5 \\
   --lr-decay-style cosine \\
   --min-lr 1e-6 \\
   --lr-warmup-fraction 0.1 \\
   --weight-decay 0.1 \\
   --adam-beta1 0.9 \\
   --adam-beta2 0.95 \\
   --use-wandb \\
   --wandb-project code-agent-dense-mainline \\
   --wandb-group dense-9b-sft-coldstart \\
   --attention-dropout 0.0 \\
   --hidden-dropout 0.0 \\
   --accumulate-allreduce-grads-in-fp32 \\
   --attention-softmax-in-fp32 \\
   --attention-backend flash
EOF
}

case "$MODE" in
  --print)
    check_pin
    echo "== SFT cold-start draft (dense Qwen3.5-9B). NOT runnable yet; see configs/slime/dense-mainline-pipeline.json =="
    echo; echo "== step 0: student mcore conversion =="; print_convert_cmd
    echo; echo "== step 1: SFT training plan =="; print_train_plan
    echo; echo "Data: ${SFT_DATA} (build with: bash -c 'python scripts/make_sft_trainset.py --max-tokens 16384')"
    ;;
  --launch)
    check_pin
    echo 'SFT launch refused: GPU stage not authorized. Record UUIDs in configs/gpu-allocation.json and review hyperparameters with the user first.' >&2
    exit 2
    ;;
  *)
    echo "Usage: bash scripts/train_sft_coldstart.sh --print | --launch" >&2
    exit 2
    ;;
esac
