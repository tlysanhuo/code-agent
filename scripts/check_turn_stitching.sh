#!/usr/bin/env bash
# Multi-turn stitching gate (the pre-launch check that was missing; incident #5).
# Drives the REAL OpenAIAdapter through a 2-turn session against vendor fakes,
# with a client that echoes assistant messages WITHOUT reasoning_content --
# exactly what DSH/pi-ai does. Acceptance for the turn-sample adapter:
#   1. >= 2 samples (one per turn);
#   2. every turn's sampled output tokens are trainable (mask sum == response
#      length == scripted output length);
#   3. total trainable tokens == total scripted output tokens.
# Also runs the VANILLA adapter on the same scenario and asserts it trains only
# the final turn -- proving the gate detects the original defect.
# Environment mirrors scripts/slime_rl.sh (CPU venv + its torch LD_LIBRARY_PATH).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$CODE_AGENT_ROOT:$CODE_AGENT_ROOT/vendor/slime"
# train venv: has sglang (the reasoning parser import) and aiohttp
exec .venv-train-rl/bin/python scripts/check_turn_stitching.py
