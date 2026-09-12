#!/usr/bin/env bash
# CPU inspection entry only. No model, trainer, controller or task launch.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# The host exports Python 3.10 Torch DSOs; do not load them into Python 3.12.
export LD_LIBRARY_PATH="$CODE_AGENT_ROOT/.venv-agl-cpu/lib/python3.12/site-packages/torch/lib"
export PYTHONPATH="$CODE_AGENT_ROOT/vendor/agent-lightning"
case "${1:---help}" in
  --compatibility-check)
    exec .venv-agl-cpu/bin/python scripts/check_agl_compatibility.py
    ;;
  --upstream-check)
    agl_run_dir="$CODE_AGENT_ROOT/runtime/agent-rl/agl-integration/check-$(date -u +%Y%m%dT%H%M%S)"
    mkdir -p "$agl_run_dir"
    exec .venv-agl-cpu/bin/python -m pytest \
      vendor/agent-lightning/tests/server vendor/agent-lightning/tests/controller \
      vendor/agent-lightning/tests/verl vendor/agent-lightning/tests/examples/test_swe_smith_agent.py \
      -q --basetemp="$agl_run_dir/tmp" -o cache_dir="$CODE_AGENT_ROOT/cache/agl-pytest" \
      --junitxml="$agl_run_dir/junit.xml"
    ;;
  *) echo 'Usage: bash scripts/agl_cpu.sh --upstream-check | --compatibility-check (expected exit 2 for current DSH streaming incompatibility)';;
esac
