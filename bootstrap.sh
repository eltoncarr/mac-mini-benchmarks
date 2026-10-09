#!/usr/bin/env bash
# One-time setup for a clean Mac mini (Apple silicon, macOS).
#
#   ./bootstrap.sh                 # everything
#   ./bootstrap.sh --no-docker     # skip Docker (SWE-bench needs it)
#   ./bootstrap.sh --no-mlx        # skip the MLX model download
#   ./bootstrap.sh --no-download   # tools only, no model downloads
#   ./bootstrap.sh --wired-limit   # raise the GPU wired-memory limit (sudo, until reboot)
#
# Safe to re-run: every step is skipped when already done.
source "$(dirname "$0")/scripts/lib.sh"

WANT_DOCKER=1 WANT_MLX=1 WANT_DOWNLOAD=1 WANT_WIRED=0
for a in "$@"; do
  case "$a" in
    --no-docker) WANT_DOCKER=0 ;;
    --no-mlx) WANT_MLX=0 ;;
    --no-download) WANT_DOWNLOAD=0 ;;
    --wired-limit) WANT_WIRED=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) die "unknown flag: $a" ;;
  esac
done

# ---- 0. Platform ------------------------------------------------------------
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "This needs an Apple silicon Mac."
log "Machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
free_gb=$(df -g "$ROOT" | awk 'NR==2{print $4}')
if (( free_gb < 120 )); then warn "Only ${free_gb} GB free. Models + SWE-bench images need ~120 GB."; fi

# ---- 1. Xcode command line tools -----------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  die "Finish the Xcode Command Line Tools install dialog, then re-run ./bootstrap.sh"
fi

# ---- 2. Homebrew + CLI tools ----------------------------------------------------
if ! command -v brew >/dev/null; then
  log "Installing Homebrew"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
log "Installing brew packages"
brew install --quiet cmake ninja git jq uv

# ---- 3. Docker (for SWE-bench) ---------------------------------------------------
if [[ $WANT_DOCKER == 1 ]]; then
  if docker info >/dev/null 2>&1; then
    log "Docker already running ($(docker version -f '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null))"
  else
    log "Installing Colima + Docker CLI (Rosetta enabled for x86_64 SWE-bench images)"
    brew install --quiet colima docker docker-buildx
    mkdir -p ~/.docker/cli-plugins
    ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx
    colima start --vm-type vz --vz-rosetta --cpu 6 --memory 12 --disk 200
  fi
fi

# ---- 4. llama.cpp (Metal) ------------------------------------------------------
mkdir -p "$VENDOR"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  log "Cloning llama.cpp"
  git clone --quiet https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --quiet --tags origin
if [[ "$LLAMA_CPP_REF" == master ]]; then
  git -C "$LLAMA_DIR" checkout --quiet master && git -C "$LLAMA_DIR" pull --quiet --ff-only
else
  git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_CPP_REF"
fi
log "Building llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null
cmake --build "$LLAMA_DIR/build" --target llama-bench llama-server llama-cli -j >/dev/null
llama_supports_mtp && log "llama-server supports MTP speculative decoding" \
  || warn "This llama.cpp build has no draft-mtp; MTP runs will be skipped"

# ---- 5. Python env ------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating Python 3.12 venv"
  uv venv --quiet --python 3.12 "$VENV"
fi
log "Installing Python packages"
VIRTUAL_ENV="$VENV" uv pip install --quiet -r "$ROOT/requirements.txt"
activate_venv

# ---- 6. Models ----------------------------------------------------------------
if [[ $WANT_DOWNLOAD == 1 ]]; then
  dest="$MODELS/${GGUF_REPO//\//__}"
  for q in $GGUF_QUANTS; do
    if [[ -n "$(gguf_path "$q")" ]]; then log "GGUF $q present"; continue; fi
    log "Downloading $GGUF_REPO [$q]"
    hf download "$GGUF_REPO" --local-dir "$dest" \
      --include "*-${q}.gguf" "*-${q}-*-of-*.gguf" "${q}/*.gguf"
    [[ -n "$(gguf_path "$q")" ]] || die "No GGUF matching $q in $GGUF_REPO; check GGUF_QUANTS"
  done
  if [[ $WANT_MLX == 1 && -n "$MLX_REPO" ]]; then
    log "Downloading $MLX_REPO"
    hf download "$MLX_REPO" --local-dir "$MODELS/mlx/${MLX_REPO//\//__}" >/dev/null
  fi
  if [[ "$RUN_4167_REFERENCE" == 1 ]]; then
    log "Downloading Llama-2 7B reference GGUFs (llama.cpp Discussion #4167)"
    hf download ggml-org/Llama-2-7B-GGUF --local-dir "$MODELS/llama2-7b" \
      --include "*[Ff]16*.gguf" "*Q8_0*.gguf" "*Q4_0*.gguf" >/dev/null
  fi
fi

# ---- 7. GPU memory limit -------------------------------------------------------
# macOS lets the GPU wire ~75% of RAM by default (~48 GB on 64 GB). Q8_0 plus a
# 128k KV cache can exceed that.
if [[ $WANT_WIRED == 1 ]]; then
  sudo sysctl iogpu.wired_limit_mb=57344
  log "GPU wired limit set to 56 GB until reboot"
fi

log "Bootstrap complete. Next: ./run.sh quick   (see README.md for suites)"
