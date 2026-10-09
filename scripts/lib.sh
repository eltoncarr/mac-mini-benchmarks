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

# Resolve the first .gguf file (first shard for split models) for a quant.
gguf_path() {
  local quant="$1" dir="$MODELS/${GGUF_REPO//\//__}"
  find "$dir" -type f \( -name "*-${quant}.gguf" -o -name "*-${quant}-00001-of-*.gguf" \) \
    ! -name "mmproj*" 2>/dev/null | sort | head -n1
}

primary_quant() { echo "${GGUF_QUANTS%% *}"; }

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

# start_llama_server <gguf> <logfile> [extra args...]; sets SERVER_PID.
start_llama_server() {
  local model="$1" logf="$2"; shift 2
  "$LLAMA_BIN/llama-server" -m "$model" --host 127.0.0.1 --port "$SERVER_PORT" \
    --alias "$MODEL_NAME" -ngl 999 -fa on --jinja --metrics \
    --cache-type-k "$KV_CACHE_TYPE" --cache-type-v "$KV_CACHE_TYPE" \
    "$@" >"$logf" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_http "http://127.0.0.1:$SERVER_PORT/health" 900; then
    tail -n 40 "$logf" >&2
    die "llama-server did not become healthy (log: $logf)"
  fi
  log "llama-server up (pid $SERVER_PID): $*"
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
