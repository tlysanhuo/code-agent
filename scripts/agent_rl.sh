#!/usr/bin/env bash
# Active architecture is slime. Historical verl checks are retained as evidence.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/slime_rl.sh" "$@"
