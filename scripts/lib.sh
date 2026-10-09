# shellcheck shell=bash
# Shared helpers. Sourced by bootstrap.sh and run.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config.env
source "$ROOT/config.env"
[[ -f "$ROOT/config.local.env" ]] && source "$ROOT/config.local.env"

VENV="$ROOT/.venv"
VENDOR="$ROOT/vendor"
MODELS="$ROOT/models"
LLAMA_DIR="$VENDOR/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/build/bin"
export HF_HOME="${HF_HOME:-$ROOT/models/.hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

activate_venv() {
  [[ -x "$VENV/bin/python" ]] || die "No venv at $VENV. Run ./bootstrap.sh first."
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

# GGUF_MODELS helpers. Each non-empty line: "label repo file-pattern".
model_labels() { awk 'NF==3{print $1}' <<<"$GGUF_MODELS"; }
model_repo()   { awk -v l="$1" 'NF==3 && $1==l{print $2}' <<<"$GGUF_MODELS"; }
model_glob()   { awk -v l="$1" 'NF==3 && $1==l{print $3}' <<<"$GGUF_MODELS"; }
model_dir()    { local r; r="$(model_repo "$1")"; echo "$MODELS/${r//\//__}"; }
# Path of the model's .gguf (first shard for split files), empty if not downloaded.
model_path() {
  find "$(model_dir "$1")" -type f -name "$(model_glob "$1")" ! -name "mmproj*" 2>/dev/null | sort | head -n1
}

llama_supports_mtp() {
  "$LLAMA_BIN/llama-server" --help 2>&1 | grep -q "draft-mtp"
}

wait_for_http() {
  local url="$1" timeout="${2:-600}" t=0
  until curl -sf "$url" >/dev/null 2>&1; do
    sleep 2; t=$((t + 2))
    (( t >= timeout )) && return 1
  done
}

# start_llama_server <gguf> <logfile> [extra args...]; sets SERVER_PID, returns 1 on failure.
start_llama_server() {
  local model="$1" logf="$2"; shift 2
  "$LLAMA_BIN/llama-server" -m "$model" --host 127.0.0.1 --port "$SERVER_PORT" \
    --alias "$MODEL_NAME" -ngl 999 -fa on --jinja --metrics \
    --cache-type-k "$KV_CACHE_TYPE" --cache-type-v "$KV_CACHE_TYPE" \
    "$@" >"$logf" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_http "http://127.0.0.1:$SERVER_PORT/health" 900; then
    tail -n 20 "$logf" >&2
    warn "llama-server did not become healthy (log: $logf)"
    stop_pid "$SERVER_PID"; SERVER_PID=""
    return 1
  fi
  log "llama-server up (pid $SERVER_PID): $*"
}

# True if the running server actually drafts with MTP (llama.cpp reports draft_n).
mtp_active() {
  local n
  n="$(curl -sf "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 40, comma separated."}],"max_tokens":96,
         "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.timings.draft_n // 0' 2>/dev/null)"
  [[ "${n:-0}" -gt 0 ]]
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
}

# Optional power sampling. start_power <outfile>; sets POWER_PID.
start_power() {
  POWER_PID=""
  [[ "${MEASURE_POWER:-0}" == 1 ]] || return 0
  sudo -n true 2>/dev/null || { warn "MEASURE_POWER=1 but sudo needs a password; run 'sudo -v' first"; return 0; }
  # shellcheck disable=SC2024  # output file is owned by the user on purpose
  sudo powermetrics --samplers cpu_power,gpu_power -i 1000 >"$1" 2>/dev/null &
  POWER_PID=$!
}
stop_power() {
  [[ -n "${POWER_PID:-}" ]] || return 0
  sudo kill "$POWER_PID" 2>/dev/null || true
  POWER_PID=""
}
