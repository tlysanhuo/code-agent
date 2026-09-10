#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
: "${UV_CACHE_DIR:?env.sh did not set project cache paths}"
command -v uv >/dev/null || { echo 'uv is required: https://docs.astral.sh/uv/' >&2; exit 2; }
command -v git >/dev/null || { echo 'git is required' >&2; exit 2; }
mkdir -p cache runtime tmp logs vendor
slime_pin='4c193f1f37509cca70f0e88807a9305b70f63f4e'
if [[ ! -d vendor/slime/.git ]]; then
  git clone --filter=blob:none https://github.com/THUDM/slime.git vendor/slime
fi
git -C vendor/slime fetch --depth=1 origin "$slime_pin"
git -C vendor/slime checkout --detach "$slime_pin"
git -C vendor/slime status --short
uv venv .venv-slime-cpu --python 3.12
uv pip sync --python .venv-slime-cpu/bin/python --index-url https://download.pytorch.org/whl/cpu configs/slime/cpu-requirements.lock
uv venv .venv-dsh --python 3.12
uv pip sync --python .venv-dsh/bin/python configs/dsh-requirements.lock
echo 'Setup complete. Run the three CPU checks from README.md.'
