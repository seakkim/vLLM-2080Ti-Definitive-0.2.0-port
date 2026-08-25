#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run.sh — Non-interactive launch wrapper for vLLM 2080 Ti Definitive v0.2.x
#
# Delegates to launcher.sh in non-interactive mode. All environment variables
# and CLI arguments are forwarded verbatim. The launcher handles:
#   - Config registry (CLI flags, env vars, profile overrides)
#   - Mode defaults, runtime defaults, profile sourcing
#   - CUDA/toolchain environment setup
#   - Argument construction, health check, smoke test
#
# Differences from the v0.1.x run.sh:
#   - No exec: launcher backgrounds the server and waits for /health + smoke.
#     Control returns to the caller after a successful launch.
#   - ACTIVE_MODEL warmup file is written before delegating (preserved).
#   - Speculative method uses "mtp" instead of "qwen3_next_mtp".
#   - Pipeline parallel (--pipeline-parallel-size) is supported via PP_SIZE.
# ---------------------------------------------------------------------------

# ---- Warmup model setup (from v0.1.x run.sh) ----

if [[ -n "${ACTIVE_MODEL:-}" ]]; then
  echo "SERVED_NAME=${SERVED_NAME:-$(basename "${MODEL_DIR:-}")}" > "$ACTIVE_MODEL"
fi

# ---- Resolve script directory ----

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- Delegate to launcher.sh ----

exec "$SCRIPT_DIR/launcher.sh" --non-interactive "$@"