#!/usr/bin/env bash
#
# Launch llama-server on the Apple GPU for one of the three experiment
# models. Kills any running llama-server first so only one model is up at a time.
# All three models run at the experiment context window of 128K (131072 tokens).
#
# Usage:
#   scripts/setup/serve_local_llm.sh <model key> [flashattn-on|flashattn-off]
#
# <model key> is a key from src/mimic_data_prep/models.py (e.g. gemma_4_e4b),
# which also supplies the server alias, the weights file, and the per-model
# flash-attention default. The optional second argument overrides that default.

set -euo pipefail

# This script lives at scripts/setup/, so the repo root is two levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_KEY="${1:-}"
model_field() {
  (cd "$REPO_ROOT" && uv run --quiet python -m mimic_data_prep.models "$MODEL_KEY" "$1")
}
if ! ALIAS="$(model_field alias)"; then
  echo "Usage: $0 <model key> [flashattn-on|flashattn-off]" >&2
  exit 2
fi
MODEL_FILE="$(model_field gguf_file)"
if [[ "$(model_field flash_attn)" == "true" ]]; then
  FLASH_ATTN="on"
else
  FLASH_ATTN="off"
fi

# Optional flash-attention override (second argument), defaulting to the
# per-model value above.
FA_OVERRIDE="${2:-}"
if [[ -n "$FA_OVERRIDE" ]]; then
  case "$FA_OVERRIDE" in
    flashattn-on) FLASH_ATTN="on" ;;
    flashattn-off) FLASH_ATTN="off" ;;
    *)
      echo "flash-attn override must be 'flashattn-on' or 'flashattn-off', got: '$FA_OVERRIDE'" >&2
      exit 2
      ;;
  esac
fi

LLAMA_SERVER="$REPO_ROOT/local-llm/llama.cpp/llama-server"
MODEL_PATH="$REPO_ROOT/local-llm/models/$MODEL_FILE"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER" >&2
  echo "Run scripts/setup/fetch_llama_and_models.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found at $MODEL_PATH" >&2
  echo "Run scripts/setup/fetch_llama_and_models.sh first." >&2
  exit 1
fi

if pgrep -f llama-server >/dev/null; then
  echo "Stopping running llama-server"
  pkill -f llama-server || true
  for _ in 1 2 3 4 5; do
    pgrep -f llama-server >/dev/null || break
    sleep 1
  done
fi

# Launch parameters, hoisted so the launch record and the server command share
# one source of truth.
CTX_SIZE=131072
N_GPU_LAYERS=999
BATCH_SIZE=2048
UBATCH_SIZE=2048
CACHE_TYPE_K=f16
CACHE_TYPE_V=f16

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${ALIAS}-$(date +%Y-%m-%d-%H%M%S).log"

# Launch record: the run manifest reads this (provenance.read_launch_record) so
# it remembers exactly how the server was launched, including flash attention,
# which /props does not fully expose. Keep the filename in sync with
# LAUNCH_RECORD_TEMPLATE in src/bhc_experiment/provenance.py.
# Record the version, build, and commit ID for reproducibility.
VERSION_LINE="$("$LLAMA_SERVER" --version 2>&1 | grep -m1 'version:')"
VERSION_PATTERN='version: ([0-9.]+)[^(]*\(build ([0-9]+), commit ([0-9a-f]+)\)'
if [[ "$VERSION_LINE" =~ $VERSION_PATTERN ]]; then
  SERVER_VERSION="${BASH_REMATCH[1]}"
  SERVER_BUILD="${BASH_REMATCH[2]}"
  SERVER_COMMIT="${BASH_REMATCH[3]}"
else
  echo "Could not read the version from: $VERSION_LINE" >&2
  exit 1
fi
RECORD_FILE="$LOG_DIR/launch-record-${ALIAS}.txt"
{
  echo "flash_attn=$FLASH_ATTN"
  echo "ctx_size=$CTX_SIZE"
  echo "n_gpu_layers=$N_GPU_LAYERS"
  echo "batch_size=$BATCH_SIZE"
  echo "ubatch_size=$UBATCH_SIZE"
  echo "cache_type_k=$CACHE_TYPE_K"
  echo "cache_type_v=$CACHE_TYPE_V"
  echo "model_file=$MODEL_FILE"
  echo "server_version=$SERVER_VERSION"
  echo "server_build=$SERVER_BUILD"
  echo "server_commit=$SERVER_COMMIT"
  echo "launched_at=$(date +%Y-%m-%dT%H:%M:%S%z)"
} > "$RECORD_FILE"

echo "Launching $ALIAS on the Apple GPU (flash-attn $FLASH_ATTN, ctx $CTX_SIZE)"
echo "Logs: $LOG_FILE"
echo "Launch record: $RECORD_FILE"

# --jinja activates the model's Jinja chat template so per-request
# chat_template_kwargs (enable_thinking) are honored. Without it, llama-server
# ignores enable_thinking and reasoning models (Gemma 4, Qwen) think anyway,
# even though every request turns reasoning off.
"$LLAMA_SERVER" \
  --model "$MODEL_PATH" --alias "$ALIAS" \
  --host 127.0.0.1 --port 8080 \
  --jinja \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --threads 6 --threads-batch 6 \
  --ctx-size "$CTX_SIZE" \
  --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
  --flash-attn "$FLASH_ATTN" \
  --cache-type-k "$CACHE_TYPE_K" --cache-type-v "$CACHE_TYPE_V" \
  --parallel 1 \
  --fit off \
  --metrics 2>&1 | tee "$LOG_FILE"
