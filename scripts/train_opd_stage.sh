#!/usr/bin/env bash
# Stage 3 (OPD) draft for the dense mainline: slime native --use-opd with the
# Qwen3.5-27B teacher as an external SGLang server (official sglang mode).
# MODE=prompt-smoke: pure OPD on the prompt pool (no task environments) - first
#   GPU validation of the teacher server + rm plumbing; scalar rewards are 0.0
#   and the teacher KL is the learning signal (official example behavior).
# MODE=agent: target state - DSH multi-turn rollouts on SWE-Gym/SWE-smith task
#   environments with test-pass GRPO rewards and OPD KL stacked on top.
# Default action PRINTS the plan; --launch is hard-gated on GPU authorization.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

ACTION="${1:---print}"
MODE="${OPD_MODE:-prompt-smoke}"   # prompt-smoke | agent

TEACHER_HF="models/Qwen3.5-27B/fc05daec18b0a78c049392ed2e771dde82bdf654"
STUDENT_HF="models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STUDENT_MCORE="${STUDENT_HF}/torch_dist"
STUDENT_LOAD="models/dense-9B-sft/slime"        # stage-1 output; must exist before this stage
OPD_DATA="data/opd-pool/opd-prompts.parquet"
TEACHER_PORT=13141
SLIME_PIN='4c193f1f37509cca70f0e88807a9305b70f63f4e'

check_pin() {
  [[ "$(git -C vendor/slime rev-parse HEAD)" == "$SLIME_PIN" ]] || { echo 'vendor/slime is not at the pinned commit' >&2; exit 2; }
  [[ -z "$(git -C vendor/slime status --porcelain)" ]] || { echo 'vendor/slime working tree is dirty' >&2; exit 2; }
}

print_teacher_server() {
  cat <<EOF
# Dedicated teacher card (record UUID in configs/gpu-allocation.json first):
CUDA_VISIBLE_DEVICES=<teacher-gpu-uuid> python3 -m sglang.launch_server \\
    --model-path ${TEACHER_HF} \\
    --host 0.0.0.0 --port ${TEACHER_PORT} \\
    --tp 1 \\
    --chunked-prefill-size 4096 \\
    --mem-fraction-static 0.8 \\
    > runtime/opd/sglang-teacher.log 2>&1 &
# note: official example uses 0.6 for Qwen3-32B (bigger-card assumption); on an 80GB
# H100 the 27B BF16 weights alone are ~54GB, so 0.8 leaves ~10GB KV cache.
# deviations from the official on_policy_distillation example (prompt-smoke mode, flagged for user review):
#   --rollout-max-prompt-len 4096: pool max is ~2.6k tokens + template headroom (example sets none)
#   --rollout-max-response-len 8192: SWE task single-turn answers (example uses 16384 for math)
#   --num-rollout 300 / batch 16 / n 4 / lr 1e-6 / opd-kl-coef 1.0: unchanged from the example
# readiness: curl -sf http://127.0.0.1:${TEACHER_PORT}/health_generate
# stop after the dependent job finishes; verify release (nvidia-smi + configs/gpu-allocation.json update)
EOF
}

print_train_plan() {
  cat <<EOF
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
   --load ${STUDENT_LOAD} \\
   --save models/dense-9B-opd/slime/ \\
   --save-interval 20 \\
   --prompt-data ${OPD_DATA} \\
   --input-key messages \\
   --apply-chat-template \\
   --rollout-shuffle \\
   --num-rollout 300 \\
   --rollout-batch-size 16 \\
   --n-samples-per-prompt 4 \\
   --rollout-max-prompt-len 4096 \\
   --rollout-max-response-len 8192 \\
   --rollout-temperature 1 \\
   --global-batch-size 64 \\
   --balance-data \\
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func \\
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards \\
   --rm-url http://127.0.0.1:${TEACHER_PORT}/generate \\
   --advantage-estimator grpo \\
   --use-opd \\
   --opd-type sglang \\
   --opd-kl-coef 1.0 \\
   --use-kl-loss \\
   --kl-loss-coef 0.00 \\
   --kl-loss-type low_var_kl \\
   --entropy-coef 0.00 \\
   --tensor-model-parallel-size 1 \\
   --sequence-parallel \\
   --pipeline-model-parallel-size 1 \\
   --recompute-granularity full \\
   --recompute-method uniform \\
   --recompute-num-layers 1 \\
   --use-dynamic-batch-size \\
   --max-tokens-per-gpu 16384 \\
   --optimizer adam \\
   --lr 1e-6 \\
   --lr-decay-style constant \\
   --weight-decay 0.1 \\
   --adam-beta1 0.9 \\
   --adam-beta2 0.98 \\
   --use-wandb \\
   --wandb-project code-agent-dense-mainline \\
   --wandb-group dense-9b-opd-${MODE} \\
   --attention-dropout 0.0 \\
   --hidden-dropout 0.0 \\
   --accumulate-allreduce-grads-in-fp32 \\
   --attention-softmax-in-fp32 \\
   --attention-backend flash
EOF
}

print_agent_mode_notes() {
  cat <<'EOF'
# AGENT MODE (target state, requires pending wiring - NOT draftable as flags yet):
#  - rollout via custom-generate-function-path slime_dsh.generate.generate (CPU-verified import/bind)
#  - task environments: SWE-Gym / SWE-smith under existing buggy/gold validation + exclusion audit
#  - rewards: pytest pass/fail per task (replaces the pure-OPD 0.0 scalar; wire via custom rm post-process)
#  - OPD unchanged: --use-opd --opd-type sglang stays stacked on the GRPO advantages
#  - SOD/TurnOPD step-wise/turn-wise reweighting = later refinement (first version runs naive OPD)
EOF
}

case "$ACTION" in
  --print)
    check_pin
    echo "== OPD stage draft (mode=${MODE}). NOT runnable yet; see configs/slime/dense-mainline-pipeline.json =="
    echo; echo "== teacher server (1 dedicated H100) =="; print_teacher_server
    echo; echo "== train plan (official on_policy_distillation example structure, project paths) =="; print_train_plan
    echo; print_agent_mode_notes
    ;;
  --launch)
    check_pin
    echo "OPD launch refused (mode=${MODE}): GPU stage not authorized. Record UUIDs in configs/gpu-allocation.json; agent mode additionally requires the DSH environment wiring." >&2
    exit 2
    ;;
  *)
    echo "Usage: OPD_MODE=prompt-smoke|agent bash scripts/train_opd_stage.sh --print | --launch" >&2
    exit 2
    ;;
esac
