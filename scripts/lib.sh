# shellcheck shell=bash
# Shared helpers. Sourced by bootstrap.sh and run.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config.env
source "$ROOT/config.env"
if [[ -f "$ROOT/config.local.env" ]]; then source "$ROOT/config.local.env"; fi

VENV="$ROOT/.venv"
VENDOR="$ROOT/vendor"
MODELS="$ROOT/models"
LLAMA_DIR="$VENDOR/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/build/bin"
export HF_HOME="${HF_HOME:-$ROOT/models/.hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1   # Python tools write to log files; show progress as it happens
# Local servers (llama-server, proxy, mlx_lm.server) must bypass any HTTP(S) proxy
# for the Python clients too (server_bench, llama-benchy, mini-swe-agent, EvalPlus).
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
# During a run, warnings and errors are also kept in $RUN/problems.log for summary.md.
warn() {
  printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2
  if [[ -n "${RUN:-}" && -d "${RUN:-}" ]]; then echo "[$(date +%H:%M:%S)] warn: $*" >>"$RUN/problems.log"; fi
}
die() {
  printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2
  if [[ -n "${RUN:-}" && -d "${RUN:-}" ]]; then echo "[$(date +%H:%M:%S)] error: $*" >>"$RUN/problems.log"; fi
  exit 1
}

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
  # --noproxy: a local server must never be reached through an HTTP(S)_PROXY
  until curl -sf --noproxy '*' "$url" >/dev/null 2>&1; do
    if (( t >= timeout )); then return 1; fi
    sleep 2; t=$((t + 2))
  done
  return 0
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

# Make sure llama-benchy's cache holds a real copy of BENCHY_BOOK_URL (it would
# otherwise download it itself and fail on an empty or HTML response).
benchy_book_path() {
  echo "$HOME/.cache/llama-benchy/$(python3 -c 'import hashlib,sys; print(hashlib.md5(sys.argv[1].encode()).hexdigest())' "$BENCHY_BOOK_URL").txt"
}

ensure_benchy_book() {
  local cache; cache="$(benchy_book_path)"
  if [[ -s "$cache" ]] && (( $(wc -c <"$cache") > 100000 )); then return 0; fi
  mkdir -p "$(dirname "$cache")"
  local tmp="$cache.tmp"
  curl -fsSL --retry 3 -A "Mozilla/5.0 (mac-mini-benchmarks)" "$BENCHY_BOOK_URL" -o "$tmp" \
    || { rm -f "$tmp"; warn "could not download $BENCHY_BOOK_URL"; return 1; }
  if (( $(wc -c <"$tmp") < 100000 )) || ! grep -q "START OF TH" "$tmp"; then
    rm -f "$tmp"; warn "$BENCHY_BOOK_URL did not return a Gutenberg plain-text book"; return 1
  fi
  # same cleanup llama-benchy applies: start at the Gutenberg START marker
  awk 'found || /\*\*\* START OF TH/ {found=1; print}' "$tmp" >"$cache" && rm -f "$tmp"
  log "llama-benchy text cached ($(wc -c <"$cache" | tr -d ' ') bytes): $cache"
}

# A leftover "credsStore" (e.g. "desktop" after switching from Docker Desktop to
# Colima) whose docker-credential-<name> helper is missing breaks every image pull.
# Prints the store name if it is broken.
docker_broken_credstore() {
  local cfg="$HOME/.docker/config.json" store
  [[ -f "$cfg" ]] || return 1
  store="$(jq -r '.credsStore // empty' "$cfg" 2>/dev/null)"
  [[ -n "$store" ]] && ! command -v "docker-credential-$store" >/dev/null && echo "$store"
}

# SWE-bench images are x86_64 only; check Docker can run them before the agent starts.
docker_amd64_ok() {
  local out
  if ! out="$(docker run --rm --platform linux/amd64 busybox:1.36 uname -m 2>&1)"; then
    warn "Docker cannot run linux/amd64 containers: $(tail -n 3 <<<"$out" | tr '\n' ' ')"
    local store; store="$(docker_broken_credstore)" || true
    if [[ -n "$store" ]]; then
      warn "$HOME/.docker/config.json uses credsStore '$store' but docker-credential-$store is not installed."
      warn "Fix: cp ~/.docker/config.json ~/.docker/config.json.bak && jq 'del(.credsStore)' ~/.docker/config.json.bak > ~/.docker/config.json"
    fi
    return 1
  fi
  if [[ "$(tail -n1 <<<"$out")" != x86_64 ]]; then
    warn "amd64 test container reported '$(tail -n1 <<<"$out")', expected x86_64"
    return 1
  fi
}

# True if the running server actually drafts with MTP (llama.cpp reports draft_n).
mtp_active() {
  local n
  n="$(curl -sf --noproxy '*' "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
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
