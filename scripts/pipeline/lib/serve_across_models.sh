# Shared server loop for running one generation command across the three local
# models. Sourced by run_all_models.sh, which passes the per-model command.
# This file starts and stops llama-server.
#
# Sourcing only defines functions and constants, with no side effects. The work
# happens in serve_across_models: launch each model's server, wait until it
# serves, run the command with --model <key> --yes appended, abort on the
# first failure, and tear the server down on exit. The teardown trap is armed
# inside serve_across_models, so sourcing and confirm_once never touch a
# running server.

# This file lives at scripts/pipeline/lib/, so the repo root is three levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVE="$REPO_ROOT/scripts/setup/serve_local_llm.sh"
LOG_DIR="$REPO_ROOT/logs"

# Must match serve_local_llm.sh's --host/--port.
SERVER_URL="http://127.0.0.1:8080"
READINESS_TIMEOUT=300 # seconds

# Timestamp each line so per-model durations are readable in a long batch.
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# One [y/N] checkpoint for the whole batch. Cancels on anything but yes, and on
# EOF (non-interactive stdin).
confirm_once() {
  local reply
  read -r -p "Proceed with these parameters? [y/N] " reply || true
  case "$reply" in
    y | Y | yes | YES) ;;
    *)
      log "Canceled"
      exit 0
      ;;
  esac
}

model_alias() {
  uv run --quiet python -m mimic_data_prep.models "$1" alias
}

_cleanup() {
  pkill -f llama-server 2>/dev/null || true
}

# Poll until the expected alias is being served, or time out. The alias check
# confirms the server loaded the requested model's weights.
_wait_for_model() {
  local alias="$1" serve_pid="$2"
  local deadline=$((SECONDS + READINESS_TIMEOUT))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      return 1 # server process died before coming up
    fi
    if curl -sf "$SERVER_URL/v1/models" 2>/dev/null | grep -qF "\"$alias\""; then
      return 0
    fi
    sleep 2
  done
  return 1 # timed out
}

# Run the given command against each model in the registry
# (src/mimic_data_prep/models.py) in turn, appending --model <key> --yes for
# each. Aborts on the first failure.
serve_across_models() {
  cd "$REPO_ROOT" # relative script paths resolve from repo root
  mkdir -p "$LOG_DIR"
  trap _cleanup EXIT # leave no server running when the batch ends or aborts

  local key alias serve_log serve_pid models
  read -ra models <<<"$(uv run --quiet python -c \
    'from mimic_data_prep.models import MODEL_KEYS; print(*MODEL_KEYS)')"
  for key in "${models[@]}"; do
    alias="$(model_alias "$key")"
    serve_log="$LOG_DIR/run-across-models-serve-${key}.log"

    log "=== ${key} (${alias}) ==="

    # Stop any running server and wait (briefly) for the port to go quiet, so
    # the readiness poll cannot match an outgoing same-alias server. Bounded so
    # a stuck port can't hang the batch.
    pkill -f llama-server 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      curl -sf "$SERVER_URL/health" >/dev/null 2>&1 || break
      sleep 1
    done

    log "Launching server (log: $serve_log)"
    "$SERVE" "$key" >"$serve_log" 2>&1 &
    serve_pid=$!

    log "Waiting up to ${READINESS_TIMEOUT}s for ${alias} to serve..."
    if ! _wait_for_model "$alias" "$serve_pid"; then
      log "Server for ${alias} did not come up. See $serve_log" >&2
      return 1
    fi
    log "${alias} is serving. Starting generation."

    "$@" --model "$key" --yes ||
      {
        log "Generation failed for ${key} (${alias}). Aborting." >&2
        return 1
      }

    log "${key} generation complete."
  done
}
