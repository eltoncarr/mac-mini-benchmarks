#!/usr/bin/env bash
# One-time setup for the vLLM (vllm-metal) harness on an Apple silicon Mac.
#
#   vllm/bootstrap.sh                 # vLLM + vllm-metal, client tools, all VLLM_MODELS
#   vllm/bootstrap.sh --model mlx-8bit # download only this model (repeatable; default: all)
#   vllm/bootstrap.sh --no-download   # software only
#   vllm/bootstrap.sh --reinstall     # rebuild the vLLM venv (e.g. after changing VLLM_METAL_VERSION)
#
# Docker for the agent suite is set up by the repo's ./bootstrap.sh (see README).
# Safe to re-run: every step is skipped when already done.
source "$(dirname "$0")/lib.sh"

WANT_DOWNLOAD=1 ONLY_MODELS="" REINSTALL=0
while (( $# )); do
  a="$1"; shift
  case "$a" in
    --model) ONLY_MODELS="$ONLY_MODELS ${1:?--model needs a label}"; shift ;;
    --no-download) WANT_DOWNLOAD=0 ;;
    --reinstall) REINSTALL=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "unknown flag: $a" ;;
  esac
done

# ---- 0. Platform ------------------------------------------------------------
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "This needs an Apple silicon Mac."
(( $(sw_vers -productVersion | cut -d. -f1) >= 15 )) || die "vllm-metal needs macOS 15 or later."
log "Machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
command -v brew >/dev/null || die "Homebrew not found; run ./bootstrap.sh (repo root) first."
brew install --quiet jq uv

# ---- 1. Client tools (repo .venv: llama-benchy, EvalPlus, mini-swe-agent, hf) ----
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating Python 3.12 venv for the client tools"
  uv venv --quiet --python 3.12 "$VENV"
fi
VIRTUAL_ENV="$VENV" uv pip install --quiet -r "$ROOT/requirements.txt"
activate_venv

# ---- 2. vLLM + vllm-metal (own venv, prebuilt wheels, pinned) ------------------
installed="$("$VLLM_BIN/python" -c 'import importlib.metadata as m; print(m.version("vllm-metal"))' 2>/dev/null || true)"
if [[ $REINSTALL == 1 || "$installed" != "$VLLM_METAL_VERSION" ]]; then
  log "Installing vLLM $VLLM_VERSION + vllm-metal $VLLM_METAL_VERSION into $VLLM_VENV"
  rm -rf "$VLLM_VENV"
  uv venv --quiet --python 3.12 "$VLLM_VENV"
  # Same two wheels vllm-metal's install.sh and Homebrew formula use, resolved together
  # so the plugin's pinned MLX / mlx-lm win.
  VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl" \
    "https://github.com/vllm-project/vllm-metal/releases/download/v${VLLM_METAL_VERSION}/vllm_metal-${VLLM_METAL_VERSION}-cp312-cp312-macosx_15_0_arm64.whl" \
    || die "vLLM/vllm-metal install failed (check VLLM_VERSION / VLLM_METAL_VERSION in vllm/config.env)"
else
  log "Keeping vllm-metal $installed (--reinstall to rebuild)"
fi
"$VLLM_BIN/vllm" --version >/dev/null 2>&1 || die "vllm CLI does not start; see: $VLLM_BIN/vllm --version"
"$VLLM_BIN/python" -c 'import vllm_metal' 2>/dev/null || die "vllm_metal plugin does not import"
log "Server stack: $(vllm_versions)"

# ---- 3. Models (MLX safetensors) ------------------------------------------------
if [[ $WANT_DOWNLOAD == 1 ]]; then
  for m in ${ONLY_MODELS:-$(vllm_labels)}; do
    [[ -n "$(vllm_repo "$m")" ]] || die "Unknown model '$m'. Labels in VLLM_MODELS: $(vllm_labels | tr '\n' ' ')"
    if vllm_model_ok "$m"; then log "Model $m present"; continue; fi
    log "Downloading $m: $(vllm_repo "$m")"
    hf download "$(vllm_repo "$m")" --local-dir "$(vllm_model_dir "$m")" >/dev/null
    vllm_model_ok "$m" || die "$(vllm_repo "$m") has no config.json + safetensors; check VLLM_MODELS"
  done
fi

log "vLLM bootstrap complete. Next: vllm/run.sh quick   (see vllm/README.md)"
