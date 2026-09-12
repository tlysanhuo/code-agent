#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mountpoint -q /lustre/prod_glm_volumes/volume-20260201002229-o7c51
test -w "$CODE_AGENT_ROOT"
df -hT / /tmp "$CODE_AGENT_ROOT"
# Do not follow a replaced environment/cache symlink outside this project.
for rl_path in .venv-train-rl cache tmp runtime logs; do
  case "$(readlink -m "$rl_path")" in
    "$CODE_AGENT_ROOT"/*) ;;
    *) echo "Path escapes project: $rl_path" >&2; exit 2 ;;
  esac
done
python - <<'PY'
import json, subprocess
manifest = json.load(open('research/sources/agent-rl-prep-20260909/source.json'))
for source in manifest['sources']:
    if source['role'].startswith('research only'):
        continue
    path = source['path']
    assert subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip() == source['commit'], path
    assert not subprocess.check_output(['git', '-C', path, 'diff', '--name-only']), path
PY
RL_UV="$CODE_AGENT_ROOT/runtime/tools/uv-package/bin/uv"
if [[ ! -f .venv-train-rl/pyvenv.cfg ]]; then
  "$RL_UV" venv --python runtime/tools/bin/python3.12 .venv-train-rl
fi
case "${1:---offline}" in
  --offline) rl_network=(--offline) ;;
  --online) rl_network=() ;;
  *) echo 'Usage: install_agent_rl.sh [--offline|--online]' >&2; exit 2 ;;
esac
export UV_HTTP_TIMEOUT=300
"$RL_UV" pip sync --python .venv-train-rl/bin/python --link-mode copy \
  "${rl_network[@]}" configs/pylock.agent-rl.toml
"$RL_UV" pip check --python .venv-train-rl/bin/python
