#!/usr/bin/env bash
# GPU training environment (local build path, user choice 2026-09-10).
# Usage: source scripts/env.sh && source scripts/env-train-gpu.sh
# Base venv: .venv-train-rl (torch 2.10.0+cu129, py3.12) + prebuilt TE 2.16.1
# (transformer_engine_cu12 wheel + meta, sha256-verified from tuna mirror).
# The system ldconfig resolves libcudart.so.12 to /usr/local/cuda-12.3 which
# shadows the venv's 12.9 runtime; the venv nvidia lib dirs must win.

CODE_AGENT_ROOT="${CODE_AGENT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
NV="$CODE_AGENT_ROOT/.venv-train-rl/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH="$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/nccl/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$CODE_AGENT_ROOT/vendor/megatron-lm-src:$CODE_AGENT_ROOT/vendor/slime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
