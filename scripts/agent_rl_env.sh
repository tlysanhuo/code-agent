#!/usr/bin/env bash
# Source only. Existing inference/DSH environment files remain untouched.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TENSORBOARD_DIR="$CODE_AGENT_ROOT/logs/agent-rl/tensorboard"
export RAY_TMPDIR="$CODE_AGENT_ROOT/tmp/agent-rl-ray"
export VLLM_RPC_BASE_PATH="$CODE_AGENT_ROOT/tmp/agent-rl-vllm"
export VLLM_CACHE_ROOT="$CODE_AGENT_ROOT/cache/agent-rl-vllm"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$CODE_AGENT_ROOT/cache/agent-rl-matplotlib"
export TORCH_EXTENSIONS_DIR="$CODE_AGENT_ROOT/cache/agent-rl-torch-extensions"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
