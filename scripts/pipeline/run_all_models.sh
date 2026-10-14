#!/usr/bin/env bash
#
# Run generation across the three local models sequentially.
#
# llama-server hosts one model at a time, so the models run in turn. The shared
# harness (lib/serve_across_models.sh) owns the server lifecycle: launch each
# model, wait until it serves, run generation, tear down. serve_local_llm.sh
# runs the server in the foreground, so it cannot be used for unattended batches.
#
# Every argument is passed to run_experiment.py for each model.
#
# Usage:
#   scripts/pipeline/run_all_models.sh --prompt brief
#   scripts/pipeline/run_all_models.sh --prompt brief --oracle-n 2
#   scripts/pipeline/run_all_models.sh --prompt unformatted --smoke

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/serve_across_models.sh"

if (($# == 0)); then
  log "Usage: $0 --prompt <variant> [--dedup none] [--oracle-n N | --smoke]" >&2
  exit 2
fi

log "Settings: $*"

confirm_once

# serve_across_models appends --model <key> --yes per model.
serve_across_models uv run python scripts/pipeline/run_experiment.py "$@"

log "All three models complete for: $*"
