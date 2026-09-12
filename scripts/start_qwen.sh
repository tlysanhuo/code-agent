#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
exec .venv-dsh/bin/python scripts/launch_qwen.py "$@"
