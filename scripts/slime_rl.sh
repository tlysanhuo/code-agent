#!/usr/bin/env bash
# Preparation and CPU checks only; never source upstream GPU launch scripts.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
case "${1:---help}" in
  --help)
    echo 'Usage: bash scripts/agent_rl.sh --status | --config | --check | --dsh-check | --harness-check | --launch'
    echo '--config prints the architecture manifest, not a runnable training recipe.'
    echo '--check runs unchanged upstream coding-agent CPU tests (fake external boundaries).'
    echo '--dsh-check uses real DSH with the upstream adapter and a scripted CPU model.'
    echo '--harness-check verifies the DSH hook in upstream task orchestration, with fake sandbox/model/evaluation boundaries.'
    echo '--launch exits 2: LoRA, task environment and GPU integration are not ready.'
    exit 0 ;;
  --status) cat configs/agent-rl/readiness.json; exit 0 ;;
  --config) cat configs/slime/architecture.json; exit 0 ;;
  --launch)
    cat configs/agent-rl/readiness.json
    echo 'Training unavailable: see missing_conditions. No Ray/GPU process started.' >&2
    exit 2 ;;
  --check|--dsh-check|--harness-check) ;;
  *) echo 'Unsupported option. The active entry now uses slime; see --help.' >&2; exit 2 ;;
esac

slime_pin='4c193f1f37509cca70f0e88807a9305b70f63f4e'
[[ "$(git -C vendor/slime rev-parse HEAD)" == "$slime_pin" ]]
[[ -z "$(git -C vendor/slime status --porcelain)" ]]
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$CODE_AGENT_ROOT/.venv-slime-cpu/lib/python3.12/site-packages/torch/lib"
export PYTHONPATH="$CODE_AGENT_ROOT:$CODE_AGENT_ROOT/vendor/slime"
if [[ "$1" == '--dsh-check' ]]; then
  exec .venv-slime-cpu/bin/python scripts/check_slime_dsh.py
fi
if [[ "$1" == '--harness-check' ]]; then
  slime_check_dir="$(mktemp -d "$CODE_AGENT_ROOT/runtime/slime/harness-check-XXXXXXXX")"
  exec .venv-slime-cpu/bin/python -m pytest scripts/check_slime_harness.py \
    -q -rs --basetemp="$slime_check_dir/tmp" \
    -o cache_dir="$CODE_AGENT_ROOT/cache/slime-harness-pytest" --junitxml="$slime_check_dir/junit.xml"
fi
slime_check_dir="$(mktemp -d "$CODE_AGENT_ROOT/runtime/slime/upstream-check-XXXXXXXX")"
exec .venv-slime-cpu/bin/python -m pytest vendor/slime/tests/test_agent \
  -q -rs --basetemp="$slime_check_dir/tmp" \
  -o cache_dir="$CODE_AGENT_ROOT/cache/slime-pytest" --junitxml="$slime_check_dir/junit.xml"
