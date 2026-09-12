#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/env.sh
uv_bin="$CODE_AGENT_ROOT/runtime/tools/uv-package/bin/uv"
if [[ ! -x .venv-swe/bin/python ]]; then
  "$uv_bin" venv --python "$CODE_AGENT_ROOT/.venv-dsh/bin/python" .venv-swe
fi
"$uv_bin" pip sync --python .venv-swe/bin/python --no-index \
  --find-links "$CODE_AGENT_ROOT/runtime/wheelhouse/swe" --require-hashes \
  configs/swe-requirements.lock
"$uv_bin" pip check --python .venv-swe/bin/python
gcc -shared -fPIC -O2 -Wall -Wextra scripts/swe_atomic_compat.c -ldl \
  -o runtime/tools/swe-atomic-compat.so
