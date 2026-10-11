#!/usr/bin/env bash
# One-time setup for a clean Mac mini (Apple silicon, macOS).
#
#   ./bootstrap.sh                 # everything
#   ./bootstrap.sh --no-docker     # skip Docker (SWE-bench needs it)
#   ./bootstrap.sh --no-mlx        # skip the MLX model download
#   ./bootstrap.sh --no-download   # tools only, no model downloads
#   ./bootstrap.sh --model q8_0    # download only this GGUF model (repeatable; default: all)
#   ./bootstrap.sh --wired-limit   # raise the GPU wired-memory limit (sudo, until reboot)
#   ./bootstrap.sh --update-llama  # pull and rebuild llama.cpp master (otherwise kept as is)
#
# Safe to re-run: every step is skipped when already done.
source "$(dirname "$0")/scripts/lib.sh"

WANT_DOCKER=1 WANT_MLX=1 WANT_DOWNLOAD=1 WANT_WIRED=0 ONLY_MODELS="" UPDATE_LLAMA=0
while (( $# )); do
  a="$1"; shift
  case "$a" in
    --model) ONLY_MODELS="$ONLY_MODELS ${1:?--model needs a label}"; shift ;;
    --no-docker) WANT_DOCKER=0 ;;
    --no-mlx) WANT_MLX=0 ;;
    --no-download) WANT_DOWNLOAD=0 ;;
    --wired-limit) WANT_WIRED=1 ;;
    --update-llama) UPDATE_LLAMA=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
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
  if store="$(docker_broken_credstore)"; then
    cp ~/.docker/config.json ~/.docker/config.json.bak
    jq 'del(.credsStore)' ~/.docker/config.json.bak >~/.docker/config.json
    log "Removed credsStore '$store' from ~/.docker/config.json (docker-credential-$store missing; backup: config.json.bak)"
  fi
  if docker run --rm --platform linux/amd64 busybox:1.36 uname -m 2>/dev/null | grep -q x86_64; then
    log "Docker runs x86_64 containers (needed for SWE-bench)"
  else
    warn "Docker cannot run x86_64 containers yet; SWE-bench needs Rosetta/QEMU emulation (see README)"
  fi
fi

# ---- 4. llama.cpp (Metal) ------------------------------------------------------
mkdir -p "$VENDOR"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  log "Cloning llama.cpp"
  git clone --quiet https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
# Keep an existing build unless asked: re-running bootstrap (e.g. to add a model)
# must not silently change the engine between benchmark passes.
if [[ -x "$LLAMA_BIN/llama-server" && $UPDATE_LLAMA == 0 && "$LLAMA_CPP_REF" == master ]]; then
  log "Keeping llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD) (--update-llama to pull master)"
else
  git -C "$LLAMA_DIR" fetch --quiet --tags origin
  if [[ "$LLAMA_CPP_REF" == master ]]; then
    git -C "$LLAMA_DIR" checkout --quiet master && git -C "$LLAMA_DIR" pull --quiet --ff-only
  else
    git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_CPP_REF"
  fi
fi
log "Building llama.cpp @ $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null
cmake --build "$LLAMA_DIR/build" --target llama-bench llama-server llama-cli -j >/dev/null
# llama-bench at the commit the #4167 thread asks for (only for ./run.sh ref4167)
if [[ "$RUN_4167_REFERENCE" == 1 && -n "$REF4167_LLAMA_REF" && ! -x "$VENDOR/llama.cpp-4167/build/bin/llama-bench" ]]; then
  log "Building llama-bench @ $REF4167_LLAMA_REF for the #4167 reference run"
  rm -rf "$VENDOR/llama.cpp-4167"
  git -C "$LLAMA_DIR" fetch --quiet origin
  git -C "$LLAMA_DIR" worktree prune
  if git -C "$LLAMA_DIR" worktree add --quiet --detach "$VENDOR/llama.cpp-4167" "$REF4167_LLAMA_REF" \
     && cmake -S "$VENDOR/llama.cpp-4167" -B "$VENDOR/llama.cpp-4167/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null \
     && cmake --build "$VENDOR/llama.cpp-4167/build" --target llama-bench -j >/dev/null; then
    log "#4167 llama-bench ready"
  else
    warn "Could not build llama-bench @ $REF4167_LLAMA_REF; ref4167 will use the main build"
  fi
fi
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
  for m in ${ONLY_MODELS:-$(model_labels)}; do
    [[ -n "$(model_glob "$m")" ]] || die "Unknown model '$m'. Labels: $(model_labels | tr '\n' ' ')"
    if [[ -n "$(model_path "$m")" ]]; then log "GGUF $m present"; continue; fi
    log "Downloading $m: $(model_repo "$m") [$(model_glob "$m")]"
    hf download "$(model_repo "$m")" --local-dir "$(model_dir "$m")" --include "$(model_glob "$m")"
    [[ -n "$(model_path "$m")" ]] || die "No file matching $(model_glob "$m") in $(model_repo "$m"); check GGUF_MODELS"
  done
  if [[ ! -f "$MODELS/tokenizer/tokenizer.json" ]]; then
    log "Downloading tokenizer from $TOKENIZER_REPO (for llama-benchy)"
    hf download "$TOKENIZER_REPO" --local-dir "$MODELS/tokenizer" \
      --include "tokenizer*" "vocab*" "merges*" "special_tokens_map.json" "config.json" "*.jinja" >/dev/null || true
  fi
  if [[ ! -f "$MODELS/tokenizer/tokenizer.json" ]]; then
    # Some Qwen repos ship no tokenizer.json; try unsloth's copy of the original weights.
    hf download "${TOKENIZER_REPO/#Qwen\//unsloth/}" --local-dir "$MODELS/tokenizer" \
      --include "tokenizer.json" "tokenizer_config.json" >/dev/null 2>&1 || true
  fi
  # Otherwise build tokenizer.json from vocab.json + merges.txt (checked against llama-server at run time).
  python "$ROOT/bench/tokenizer_fix.py" ensure "$MODELS/tokenizer" || die "No usable tokenizer in $MODELS/tokenizer"
  if [[ $WANT_MLX == 1 && -n "$MLX_REPO" ]]; then
    log "Downloading $MLX_REPO"
    hf download "$MLX_REPO" --local-dir "$MODELS/mlx/${MLX_REPO//\//__}" >/dev/null
  fi
  if [[ "$RUN_4167_REFERENCE" == 1 ]]; then
    log "Downloading Llama-2 7B reference GGUFs (llama.cpp Discussion #4167)"
    python "$ROOT/bench/ref_models.py" "$MODELS/llama2-7b" || warn "Llama-2 7B reference download incomplete"
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
