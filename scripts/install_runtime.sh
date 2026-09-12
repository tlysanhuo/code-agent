#!/usr/bin/env bash
# Rebuild the pinned environments from the prepared, hash-checked wheelhouse.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mountpoint -q "$(dirname -- "$CODE_AGENT_ROOT")"
test -w "$CODE_AGENT_ROOT"
df -hT / /tmp "$CODE_AGENT_ROOT"
project_uv="$CODE_AGENT_ROOT/runtime/tools/uv-package/bin/uv"
test -x "$project_uv"
"$project_uv" python install 3.12.12
for name in dsh runtime download; do
  env_path="$CODE_AGENT_ROOT/.venv-$name"
  if [ ! -x "$env_path/bin/python" ]; then
    "$project_uv" venv --python 3.12.12 "$env_path"
  fi
  requirements_name="$name"
  if [ "$name" = download ]; then requirements_name=downloader; fi
  "$project_uv" pip sync --python "$env_path/bin/python" --no-index \
    --find-links "$CODE_AGENT_ROOT/runtime/wheelhouse" --require-hashes \
    "$CODE_AGENT_ROOT/configs/$requirements_name-requirements.lock"
  "$project_uv" pip check --python "$env_path/bin/python"
done
