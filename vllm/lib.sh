# shellcheck shell=bash
# vLLM helpers on top of scripts/lib.sh. Sourced by vllm/bootstrap.sh and vllm/run.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/lib.sh"
# shellcheck source=config.env
source "$ROOT/vllm/config.env"
if [[ -f "$ROOT/config.local.env" ]]; then source "$ROOT/config.local.env"; fi

# The server gets its own venv (vLLM + vllm-metal + its pinned MLX/mlx-lm); the clients
# (llama-benchy, EvalPlus, mini-swe-agent, collect.py) stay in the repo's .venv.
VLLM_VENV="$VENDOR/vllm-metal-venv"
VLLM_BIN="$VLLM_VENV/bin"

# VLLM_MODELS helpers. Each non-empty line: "label repo".
vllm_labels() { awk 'NF==2{print $1}' <<<"$VLLM_MODELS"; }
vllm_repo()   { awk -v l="$1" 'NF==2 && $1==l{print $2}' <<<"$VLLM_MODELS"; }
# Same folder as the llama.cpp harness's MLX download, so mlx-4bit is shared with ./run.sh mlx.
vllm_model_dir() { local r; r="$(vllm_repo "$1")"; echo "$MODELS/mlx/${r//\//__}"; }
vllm_model_ok()  { [[ -f "$(vllm_model_dir "$1")/config.json" ]] && compgen -G "$(vllm_model_dir "$1")/*.safetensors" >/dev/null; }

# Versions of the server stack, as JSON (for env.json).
vllm_versions() {
  "$VLLM_BIN/python" - <<'PY' 2>/dev/null || echo '{}'
import importlib.metadata as m, json
out = {}
for p in ["vllm", "vllm-metal", "mlx", "mlx-lm", "mlx-metal", "torch", "transformers"]:
    try:
        out[p] = m.version(p)
    except m.PackageNotFoundError:
        pass
print(json.dumps(out))
PY
}
