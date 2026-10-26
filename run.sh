#!/usr/bin/env bash
# Run benchmark suites and collect results into results/<run>/.
#
#   ./run.sh quick      # ~30-90 min: llama-bench, llama-benchy, 1 SWE-bench task
#   ./run.sh final      # ~4-5 h: publishable set: full depth sweep, server suite, HumanEval+
#   ./run.sh standard   # overnight+: full depth sweeps, prefix cache, concurrency,
#                       #   EvalPlus, SWE-bench Verified Mini (50)
#   ./run.sh full       # days: standard + SWE-bench Verified (500) + MBPP+
#
#   Individual suites: perf | server | coding | agent | tbench [smoke|all]
#   Machine-level (no GGUF model, run once): ref4167 | mlx (macOS) | tbench-check
#
# Runs on macOS (Apple silicon, Metal) and on Linux with an NVIDIA GPU (CUDA, e.g. DGX
# Spark); the suites, models and settings are the same on both.
#   RUN_DIR=results/<run> ./run.sh rescore      # re-score saved EvalPlus samples
#   ./run.sh collect results/<run>               # rebuild summary.md
#   ./run.sh compare results/<run> results/<run>  # side-by-side table -> results/compare-*.md
#
# One pass benchmarks one model: the first in GGUF_MODELS (config.env), or
#   MODEL=q8_0 ./run.sh quick
# RUN_DIR=results/<run> ./run.sh agent  adds a suite to an existing run.
source "$(dirname "$0")/scripts/lib.sh"
activate_venv

SUITE="${1:-quick}"
if [[ "$SUITE" == collect ]]; then python "$ROOT/bench/collect.py" "${2:?run dir}"; exit; fi
if [[ "$SUITE" == compare ]]; then
  shift; (( $# >= 1 )) || die "usage: ./run.sh compare results/<run> [results/<run> ...]"
  python "$ROOT/bench/collect.py" --compare "$ROOT/results/compare-$(date +%Y%m%d-%H%M).md" "$@"; exit
fi

# rescore only re-reads files of an existing run: no model, no servers, env.json untouched
REUSE_RUN=0
if [[ "$SUITE" == rescore ]]; then
  [[ -n "${RUN_DIR:-}" && -f "$RUN_DIR/env.json" ]] || die "usage: RUN_DIR=results/<existing run> ./run.sh rescore"
  REUSE_RUN=1 MEASURE_POWER=0
fi

MODEL="${MODEL:-$(model_labels | head -n1)}"
if (( ! REUSE_RUN )); then
  [[ -n "$(model_glob "$MODEL")" ]] || die "Unknown MODEL '$MODEL'. Labels in GGUF_MODELS: $(model_labels | tr '\n' ' ')"
  [[ -n "$(model_path "$MODEL")" ]] || die "Model $MODEL not downloaded; run ./bootstrap.sh"
fi
MODELS_SEL="$MODEL"

STAMP="$(date +%Y%m%d-%H%M)"
case "$SUITE" in  # these don't use the GGUF model
  ref4167|mlx|tbench-check) RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${SUITE}}" ;;
  *)           RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${MODEL_NAME}_${MODEL}_${SUITE}}" ;;
esac
RUN="$(mkdir -p "$RUN" && cd "$RUN" && pwd)"
mkdir -p "$RUN"/{llama_bench,benchy,server,mlx,evalplus,swebench,tbench,power,logs}
SERVER_PID="" PROXY_PID="" MLX_PID="" POWER_PID=""

# Always stop servers and write summary.md, also when a step dies part-way.
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if (( rc != 0 )); then
    echo "[$(date +%H:%M:%S)] error: run stopped early (exit $rc; Ctrl-C or a fatal error); later suites did not run" >>"$RUN/problems.log"
  fi
  stop_power; stop_pid "$PROXY_PID"; stop_pid "$SERVER_PID"; stop_pid "$MLX_PID"
  python "$ROOT/bench/collect.py" "$RUN" >/dev/null 2>&1 || true
  if [[ -s "$RUN/problems.log" ]]; then
    warn "Run finished with problems (rc=$rc); see $RUN/problems.log and $RUN/logs/"
  fi
  log "Summary: $RUN/summary.md"
  exit "$rc"
}
trap cleanup EXIT INT TERM
keep_awake   # for the whole run
if [[ "${MEASURE_POWER:-0}" == 1 ]] && is_mac; then
  log "MEASURE_POWER=1: powermetrics needs sudo for the whole run"
  sudo -v || die "sudo is required for MEASURE_POWER=1"
  while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 60; done &   # keep sudo authorized
fi

MTP_ARGS=()
if (( REUSE_RUN )); then
  :   # no servers in this suite
elif [[ "$USE_MTP" == 1 ]] && llama_supports_mtp; then
  MTP_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$MTP_DRAFT_N")
elif [[ "$USE_MTP" == 1 ]]; then
  warn "This llama.cpp build has no draft-mtp; all MTP runs are skipped"
fi

if [[ "$SUITE" == ref4167 || "$SUITE" == mlx || "$SUITE" == tbench-check ]]; then MODELS_SEL=""; fi
if (( ! REUSE_RUN )); then
models_json="$(for m in $MODELS_SEL; do f="$(model_path "$m")"
  jq -nc --arg l "$m" --arg r "$(model_repo "$m")" --arg f "$(basename "$f")" --arg b "$(file_bytes "$f")" \
    '{label:$l, repo:$r, file:$f, bytes:($b|tonumber)}'; done | jq -sc .)"
python "$ROOT/bench/sysinfo.py" "$RUN/env.json" "$LLAMA_DIR" "$MODEL_NAME" \
  "$(jq -nc --arg s "$SUITE" --argjson models "$models_json" --arg kv "$KV_CACHE_TYPE" \
      --arg ctx "$SERVER_CTX" --arg mtp "${MTP_ARGS[*]:-}" \
      '{suite:$s, models:$models, kv_cache:$kv, server_ctx:($ctx|tonumber), mtp_args:$mtp}')"
fi
log "Run directory: $RUN"

# ---- perf: llama-bench pp512/tg128 + context-depth sweep ------------------------
suite_perf() {  # <depths>
  local depths="$1" m
  for m in $MODELS_SEL; do
    log "llama-bench $m (depths $depths)"
    start_power "$RUN/power/llama-bench-$m.txt"
    "$LLAMA_BIN/llama-bench" -m "$(model_path "$m")" -p 512 -n 128 -d "$depths" -ngl 99 -fa 1 \
      -r "$BENCH_REPS" -o json >"$RUN/llama_bench/$m.json" 2>"$RUN/logs/llama-bench-$m.log" \
      || warn "llama-bench $m failed (see logs/llama-bench-$m.log)"
    stop_power
  done
}

# ---- ref4167: exact command from llama.cpp Discussion #4167 ----------------------
suite_ref4167() {
  local d="$MODELS/llama2-7b" args=()
  local kind f
  for kind in F16 Q8_0 Q4_0; do   # same selection rule as bench/ref_models.py
    f="$(python -c 'import sys, pathlib; sys.path.insert(0, sys.argv[1]); import ref_models as r
fs = [str(p) for p in pathlib.Path(sys.argv[2]).rglob("*.gguf")]; print(r.pick(fs, sys.argv[3]) or "")' \
      "$ROOT/bench" "$d" "$kind")"
    if [[ -n "$f" ]]; then args+=(-m "$f"); else warn "#4167: no Llama-2 7B $kind model in $d (./bootstrap.sh)"; fi
  done
  (( ${#args[@]} == 6 )) || { warn "Llama-2 7B reference models incomplete; skipping #4167 run"; return; }
  local bench="$VENDOR/llama.cpp-4167/build/bin/llama-bench"
  if [[ -z "$REF4167_LLAMA_REF" || ! -x "$bench" ]]; then
    bench="$LLAMA_BIN/llama-bench"
    [[ -n "$REF4167_LLAMA_REF" ]] && warn "#4167: llama-bench @ $REF4167_LLAMA_REF not built (./bootstrap.sh); using the main build"
  fi
  local delay=(); "$bench" --help 2>&1 | grep -q -- "--delay" && delay=(--delay 30)
  log "llama-bench Llama-2 7B reference (Discussion #4167) with $bench"
  "$bench" ${args[@]+"${args[@]}"} -p 512 -n 128 -ngl 99 ${delay[@]+"${delay[@]}"} -o md \
    >"$RUN/llama_bench/4167.md" 2>"$RUN/logs/ref4167.log" || warn "reference run failed"
}

# ---- server: llama-benchy (pp/tg at depth, TTFT, concurrency, prefix cache) ----
#      + server_bench.py (short-chat inter-token latency percentiles)
TOKENIZER="$MODELS/tokenizer"

run_benchy() {  # <label> <port> <served-model-name> <tokenizer> <depths> [extra llama-benchy args...]
  local label="$1" port="$2" served="$3" tok="$4" depths="$5"; shift 5
  # llama-server (not mlx_lm.server) can count tokens itself: make sure llama-benchy's
  # local tokenizer agrees, otherwise its prompt sizes would be wrong.
  if [[ "$port" == "$SERVER_PORT" ]] && ! python "$ROOT/bench/tokenizer_fix.py" check "$tok" \
       --server "http://127.0.0.1:$port" --text "$(benchy_book_path)" >>"$RUN/logs/benchy-$label.log" 2>&1; then
    warn "llama-benchy $label skipped: local tokenizer disagrees with llama-server (logs/benchy-$label.log)"
    return 0
  fi
  log "llama-benchy $label (depths $depths)"
  # shellcheck disable=SC2086  # word-split the space-separated lists
  llama-benchy --base-url "http://127.0.0.1:$port/v1" --model "$MODEL_NAME" --served-model-name "$served" \
    --tokenizer "$tok" --book-url "$BENCHY_BOOK_URL" --pp $BENCHY_PP --tg "$BENCHY_TG" --depth $depths --runs "$LATENCY_REPS" \
    --latency-mode generation --exact-tg --extra-body 'chat_template_kwargs={"enable_thinking":false}' \
    --format json --save-result "$RUN/benchy/$label.json" "$@" \
    >>"$RUN/logs/benchy-$label.log" 2>&1 || warn "llama-benchy $label failed (logs/benchy-$label.log)"
}

run_latency() {  # <label> <port> <served-model-name>
  python "$ROOT/bench/server_bench.py" --base-url "http://127.0.0.1:$2/v1" --model "$3" --label "$1" \
    --reps "$LATENCY_REPS" --out "$RUN/server/$1.json" || warn "server latency $1 failed"
}

# start_model_server <model> <tag> <mtp:0|1> [extra llama-server args...]
# With mtp=1, succeeds only if the server really drafts with MTP for this model.
start_model_server() {
  local m="$1" tag="$2" mtp="$3"; shift 3
  local logf="$RUN/logs/server-$m-$tag.log"
  if [[ $mtp == 1 ]]; then
    (( ${#MTP_ARGS[@]} )) || return 1
    start_llama_server "$(model_path "$m")" "$logf" "$@" "${MTP_ARGS[@]}" || return 1
    if ! mtp_active; then
      warn "$m: no MTP drafting (GGUF has no MTP head?); MTP runs skipped for this model"
      stop_pid "$SERVER_PID"; SERVER_PID=""
      return 1
    fi
  else
    start_llama_server "$(model_path "$m")" "$logf" "$@" || return 1
  fi
}

# Serve a model for quality suites: with MTP when it works (faster), else plain.
# Sets SERVED_MTP=1/0.
start_quality_server() {
  local m="$1" tag="$2"; shift 2
  SERVED_MTP=1
  start_model_server "$m" "$tag-mtp" 1 "$@" && return 0
  SERVED_MTP=0
  start_model_server "$m" "$tag" 0 "$@"
}

suite_server() {  # <depths> <full: 1 adds prefix-cache + concurrency runs>
  local depths="$1" full="$2" m
  # Thinking off at the server, so every request (incl. llama-benchy's warmup and
  # coherence check, which don't get --extra-body) answers directly.
  local nothink=(--chat-template-kwargs '{"enable_thinking": false}')
  python "$ROOT/bench/tokenizer_fix.py" ensure "$TOKENIZER" >/dev/null \
    || die "No usable tokenizer in $TOKENIZER; run ./bootstrap.sh"
  ensure_benchy_book || warn "llama-benchy runs will likely fail (no prompt text)"
  for m in $MODELS_SEL; do
    if start_model_server "$m" plain 0 -c "$SERVER_CTX" -np 1 "${nothink[@]}"; then
      start_power "$RUN/power/server-$m.txt"
      run_benchy "llama.cpp-$m" "$SERVER_PORT" "$MODEL_NAME" "$TOKENIZER" "$depths" --no-cache
      run_latency "llama.cpp-$m" "$SERVER_PORT" "$MODEL_NAME"
      stop_power
      if [[ $full == 1 ]]; then
        run_benchy "llama.cpp-$m-prefix-cache" "$SERVER_PORT" "$MODEL_NAME" "$TOKENIZER" "4096 16384 32768" \
          --enable-prefix-caching
      fi
      stop_pid "$SERVER_PID"; SERVER_PID=""
    fi
    if start_model_server "$m" mtp 1 -c "$SERVER_CTX" -np 1 "${nothink[@]}"; then
      start_power "$RUN/power/server-$m-mtp.txt"
      run_benchy "llama.cpp-$m-mtp" "$SERVER_PORT" "$MODEL_NAME" "$TOKENIZER" "$depths" --no-cache
      run_latency "llama.cpp-$m-mtp" "$SERVER_PORT" "$MODEL_NAME"
      stop_power; stop_pid "$SERVER_PID"; SERVER_PID=""
    fi
    if [[ $full == 1 ]]; then
      local np; np="$(echo "$CONCURRENCY_LEVELS" | tr ' ' '\n' | sort -n | tail -n1)"
      if start_model_server "$m" "np$np" 0 -c $((np * 16384)) -np "$np" "${nothink[@]}"; then
        # shellcheck disable=SC2086
        run_benchy "llama.cpp-$m-concurrency" "$SERVER_PORT" "$MODEL_NAME" "$TOKENIZER" "0 4096" \
          --no-cache --concurrency $CONCURRENCY_LEVELS
        stop_pid "$SERVER_PID"; SERVER_PID=""
      fi
    fi
  done
}

# ---- mlx: mlx_lm.benchmark + the same server suite against mlx_lm.server ---------
suite_mlx() {
  is_mac || { warn "MLX runs only on Apple silicon; skipping"; return; }
  local m="$MODELS/mlx/${MLX_REPO//\//__}"
  [[ -n "$MLX_REPO" && -d "$m" ]] || { warn "MLX model not downloaded; skipping"; return; }
  if command -v mlx_lm.benchmark >/dev/null; then
    for p in 512 16384; do
      log "mlx_lm.benchmark p=$p"
      mlx_lm.benchmark --model "$m" -p "$p" -g 128 -n "$BENCH_REPS" >"$RUN/mlx/pp${p}_tg128.txt" 2>&1 \
        || warn "mlx_lm.benchmark failed (see mlx/pp${p}_tg128.txt)"
    done
  fi
  mlx_lm.server --model "$m" --host 127.0.0.1 --port "$MLX_PORT" >"$RUN/logs/mlx-server.log" 2>&1 &
  MLX_PID=$!
  ensure_benchy_book || warn "llama-benchy runs will likely fail (no prompt text)"
  if wait_for_http "http://127.0.0.1:$MLX_PORT/v1/models" 600; then
    # mlx_lm.server has no prompt cache opt-out; llama-benchy still makes every request unique.
    run_benchy "mlx-4bit" "$MLX_PORT" "$m" "$m" "0 4096 16384 32768" --no-cache
    run_latency "mlx-4bit" "$MLX_PORT" "$m"
  else
    warn "mlx_lm.server did not start (logs/mlx-server.log)"
  fi
  stop_pid "$MLX_PID"; MLX_PID=""
}

# ---- coding: EvalPlus HumanEval+/MBPP+ --------------------------------------------
# macOS rejects EvalPlus's RLIMIT_AS memory cap ("current limit exceeds maximum limit"),
# which made every test fail (0%). -1 disables only that cap; per-test timeouts remain.
export EVALPLUS_MAX_MEMORY_BYTES=-1

suite_coding() {  # <datasets>
  local datasets="$1" m ds
  for m in $MODELS_SEL; do
    start_quality_server "$m" coding -c 32768 -np 1 \
      --chat-template-kwargs "{\"enable_thinking\": $([[ $EVALPLUS_THINKING == 1 ]] && echo true || echo false)}" \
      || { warn "skip EvalPlus for $m"; continue; }
    mkdir -p "$RUN/evalplus/$m"
    for ds in $datasets; do
      log "EvalPlus $ds on $m (mtp=$SERVED_MTP)"
      OPENAI_API_KEY=local evalplus.evaluate --model "$MODEL_NAME-$m" --dataset "$ds" --backend openai \
        --base-url "http://127.0.0.1:$SERVER_PORT/v1" --greedy --root "$RUN/evalplus/raw/$m" \
        2>&1 | tee "$RUN/evalplus/$m/$ds.log" | tail -n 8 || warn "EvalPlus $ds on $m failed"
    done
    stop_pid "$SERVER_PID"; SERVER_PID=""
  done
}

# Re-score saved EvalPlus samples without regenerating (e.g. after a scoring failure).
suite_rescore() {
  local f m ds n=0
  for f in "$RUN"/evalplus/raw/*/*/*.jsonl; do
    [[ -f "$f" && "$f" != *.raw.jsonl ]] || continue
    m="$(basename "$(dirname "$(dirname "$f")")")"; ds="$(basename "$(dirname "$f")")"
    log "Re-scoring EvalPlus $ds for $m: $f"
    mkdir -p "$RUN/evalplus/$m"
    # keep the old results aside; EvalPlus would otherwise reuse them (or prompt to overwrite)
    local old="${f%.jsonl}_eval_results.json"
    if [[ -f "$old" ]]; then mv "$old" "$old.$(date +%Y%m%d-%H%M%S).bak"; fi
    if [[ -f "$RUN/evalplus/$m/$ds.log" ]]; then mv "$RUN/evalplus/$m/$ds.log" "$RUN/evalplus/$m/$ds.log.bak"; fi
    evalplus.evaluate --dataset "$ds" --samples "$f" </dev/null \
      2>&1 | tee "$RUN/evalplus/$m/$ds.log" | tail -n 4 || warn "EvalPlus re-score $ds for $m failed"
    n=$((n + 1))
  done
  (( n )) || warn "no EvalPlus samples under $RUN/evalplus/raw/"
}

# ---- agent: SWE-bench with mini-swe-agent (real repos, fix-a-bug tasks) ----------
suite_agent() {  # <smoke|mini|verified>
  local subset="$1" ds args=() m
  docker info >/dev/null 2>&1 || die "Docker is not running (needed for SWE-bench). Start Docker/Colima."
  if ! docker_amd64_ok; then
    warn "Enable x86_64 emulation: Docker Desktop > Settings > General > 'Use Rosetta for x86_64/amd64 emulation',"
    warn "or recreate Colima with: colima delete && colima start --vm-type vz --vz-rosetta --cpu 6 --memory 12 --disk 200"
    die "SWE-bench skipped: Docker cannot run the x86_64 SWE-bench images"
  fi
  case "$subset" in
    smoke)    ds=verified; args=(--filter "^${SWE_SMOKE_INSTANCE}\$") ;;
    mini)     ds=MariusHobbhahn/swe-bench-verified-mini ;;
    verified) ds=verified ;;
    *) die "unknown SWE subset $subset" ;;
  esac
  if [[ -n "$SWE_SLICE" && "$subset" != smoke ]]; then args+=(--slice "$SWE_SLICE"); fi
  local np="$SWE_WORKERS"
  for m in $MODELS_SEL; do
    # one folder per model and subset, so a later slice never mixes with the smoke task
    local tag="$subset${SWE_SLICE:+_${SWE_SLICE/:/-}}"
    [[ "$subset" == smoke ]] && tag=smoke
    local d="$RUN/swebench/$m/$tag"
    mkdir -p "$d"
    start_quality_server "$m" agent -c $((SERVER_CTX * np)) -np "$np" || { warn "skip SWE-bench for $m"; continue; }
    python "$ROOT/bench/telemetry_proxy.py" --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
      --log "$d/calls.jsonl" & PROXY_PID=$!
    python "$ROOT/bench/swe_config.py" --api-base "http://127.0.0.1:$PROXY_PORT/v1" --model "$MODEL_NAME-$m" \
      --step-limit "$SWE_STEP_LIMIT" --thinking "$AGENT_THINKING" \
      --max-tokens "$AGENT_MAX_TOKENS" --timeout "$AGENT_TIMEOUT_S" --out "$d/mini-swe-agent.yaml"
    log "mini-swe-agent on SWE-bench [$subset] with $m (mtp=$SERVED_MTP, $np worker(s))"
    local t0=$SECONDS
    MSWEA_COST_TRACKING=ignore_errors mini-extra swebench --subset "$ds" --split test \
      -c "$d/mini-swe-agent.yaml" -w "$np" -o "$d/trajs" ${args[@]+"${args[@]}"} \
      >"$RUN/logs/mini-swe-agent-$m-$tag.log" 2>&1 || warn "mini-swe-agent exited non-zero (logs/mini-swe-agent-$m-$tag.log)"
    local wall=$((SECONDS - t0))
    stop_pid "$PROXY_PID"; PROXY_PID=""; stop_pid "$SERVER_PID"; SERVER_PID=""
    jq -n --arg model "$m" --arg s "$subset${SWE_SLICE:+[$SWE_SLICE]}" --arg ds "$ds" --argjson sl "$SWE_STEP_LIMIT" --argjson th "$AGENT_THINKING" \
       --argjson w "$wall" --argjson np "$np" --argjson mtp "$SERVED_MTP" \
       --arg v "$(python -c 'import importlib.metadata as m; print(m.version("mini-swe-agent"))' 2>/dev/null)" \
       '{model:$model, subset:$s, dataset:$ds, step_limit:$sl, thinking:($th==1), mtp:($mtp==1), wall_s:$w, workers:$np, mini_swe_agent:$v}' \
       >"$d/meta.json"
    if [[ -f "$d/trajs/preds.json" ]]; then
      cp "$d/trajs/preds.json" "$d/preds.json"
      swe_evaluate "$d" "$m-$tag"
    else
      warn "no preds.json produced for $m"
    fi
  done
}

swe_evaluate() {
  local d="$1" m="$2" run_id; run_id="$(basename "$RUN" | tr -c 'A-Za-z0-9_.\n-' '_')_$2"
  case "$SWE_EVAL" in
    none) log "SWE_EVAL=none: predictions saved, not evaluated" ;;
    sbcli)
      [[ -n "${SWEBENCH_API_KEY:-}" ]] || { warn "SWEBENCH_API_KEY not set; skipping sb-cli"; return; }
      log "Submitting to SWE-bench cloud evaluator (sb-cli)"
      mkdir -p "$d/sbcli"
      sb-cli submit swe-bench_verified test --predictions_path "$d/preds.json" --run_id "$run_id" \
        --output_dir "$d/sbcli" >"$RUN/logs/sbcli-$m.log" 2>&1 || warn "sb-cli failed (logs/sbcli-$m.log)"
      local r; r="$(grep -l resolved_instances "$d"/sbcli/*.json 2>/dev/null | head -n1)"
      if [[ -n "$r" ]]; then cp "$r" "$d/report.json"; fi ;;
    local)
      log "Evaluating patches with the SWE-bench harness (Docker)"
      local ids; ids=$(jq -r 'keys[]' "$d/preds.json" | tr '\n' ' ')
      # Uses the published x86_64 images (already pulled by the agent run), run under Rosetta.
      (cd "$d" && python -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Verified --split test \
        --predictions_path "$d/preds.json" --instance_ids $ids \
        --max_workers 4 --run_id "$run_id" --report_dir "$d") \
        >"$RUN/logs/swebench-eval-$m.log" 2>&1 || warn "SWE-bench evaluation failed (logs/swebench-eval-$m.log)"
      local r; r="$(ls "$d"/*."$run_id".json 2>/dev/null | head -n1)"
      if [[ -n "$r" ]]; then mv "$r" "$d/report.json"; fi ;;
  esac
}

# ---- tbench: Terminal-Bench 2.0 with Terminus-2, through Harbor -------------------
# The agent runs in the Harbor process on this machine and drives a tmux shell inside
# each task's container; its model calls go through the telemetry proxy to llama-server.
harbor_ready() {
  [[ -x "$HARBOR_VENV/bin/harbor" ]] || die "Harbor not installed in $HARBOR_VENV; run ./bootstrap.sh"
  docker info >/dev/null 2>&1 || die "Docker is not running (needed for Terminal-Bench)"
  docker compose version >/dev/null 2>&1 || die "docker compose is missing (needed for Terminal-Bench); see README"
}

harbor_env_args() {  # flags shared by the model run and the oracle check
  HARBOR_ARGS=(--dataset "$TB_DATASET" --n-attempts "$TB_ATTEMPTS" --timeout-multiplier "$TB_TIMEOUT_MULTIPLIER" --quiet)
  if [[ "$TB_NATIVE_BUILD" == 1 ]]; then HARBOR_ARGS+=(--force-build); fi
}

suite_tbench() {  # <smoke|all>
  local scope="$1" m g filt=()
  harbor_ready
  case "$scope" in
    smoke) filt=(--include-task-name "*$TB_SMOKE_TASK") ;;
    all)   for g in $TB_TASKS; do filt+=(--include-task-name "$g"); done ;;
    *) die "unknown Terminal-Bench scope $scope (smoke|all)" ;;
  esac
  harbor_env_args
  local np="$TB_CONCURRENT"
  # Qwen3.8 recommended sampling, as for SWE-bench (see bench/swe_config.py)
  local temp=1.0 call='{"top_p":0.95,"extra_body":{"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}}'
  if [[ $TB_THINKING != 1 ]]; then
    temp=0.7 call='{"top_p":0.8,"presence_penalty":1.5,"extra_body":{"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}}'
  fi
  for m in $MODELS_SEL; do
    local tag="$scope${TB_TASKS:+_$(tr -c 'A-Za-z0-9_.-' _ <<<"$TB_TASKS")}"
    [[ "$scope" == smoke ]] && tag=smoke
    local d="$RUN/tbench/$m/$tag"
    mkdir -p "$d"
    start_quality_server "$m" tbench -c $((SERVER_CTX * np)) -np "$np" || { warn "skip Terminal-Bench for $m"; continue; }
    python "$ROOT/bench/telemetry_proxy.py" --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
      --log "$d/calls.jsonl" & PROXY_PID=$!
    log "Terminal-Bench [$scope] with $m ($TB_AGENT, mtp=$SERVED_MTP, $np at a time, timeouts x$TB_TIMEOUT_MULTIPLIER)"
    local t0=$SECONDS
    OPENAI_API_KEY=local "$HARBOR_VENV/bin/harbor" run "${HARBOR_ARGS[@]}" --agent "$TB_AGENT" \
      --model "openai/$MODEL_NAME-$m" \
      --agent-kwarg "api_base=http://127.0.0.1:$PROXY_PORT/v1" \
      --agent-kwarg "model_info={\"max_input_tokens\":$SERVER_CTX,\"max_output_tokens\":$AGENT_MAX_TOKENS,\"input_cost_per_token\":0,\"output_cost_per_token\":0}" \
      --agent-kwarg "temperature=$temp" --agent-kwarg "llm_call_kwargs=$call" \
      --n-concurrent "$np" --jobs-dir "$d" --job-name job ${filt[@]+"${filt[@]}"} </dev/null \
      >"$RUN/logs/harbor-$m-$tag.log" 2>&1 || warn "harbor exited non-zero (logs/harbor-$m-$tag.log)"
    local wall=$((SECONDS - t0))
    stop_pid "$PROXY_PID"; PROXY_PID=""; stop_pid "$SERVER_PID"; SERVER_PID=""
    jq -n --arg model "$m" --arg s "$scope" --arg ds "$TB_DATASET" --arg ag "$TB_AGENT" --arg tasks "$TB_TASKS" \
       --argjson th "$TB_THINKING" --argjson w "$wall" --argjson np "$np" --argjson mtp "$SERVED_MTP" \
       --argjson k "$TB_ATTEMPTS" --argjson tm "$TB_TIMEOUT_MULTIPLIER" --argjson nb "$TB_NATIVE_BUILD" \
       --arg v "$("$HARBOR_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("harbor"))' 2>/dev/null)" \
       --arg arch "$(docker version -f '{{.Server.Arch}}' 2>/dev/null)" \
       '{model:$model, scope:$s, dataset:$ds, agent:$ag, tasks:$tasks, thinking:($th==1), mtp:($mtp==1), wall_s:$w,
         concurrent:$np, attempts:$k, timeout_multiplier:$tm, native_build:($nb==1), docker_arch:$arch, harbor:$v}' \
       >"$d/meta.json"
  done
}

# Run every task's reference solution (Harbor's oracle agent, no model) on this
# machine's images: tasks that fail here fail for every model, which matters with
# native arm64 builds. Run once per machine; summary.md lists the failures.
suite_tbench_check() {
  harbor_ready; harbor_env_args
  local d="$RUN/tbench/oracle/all"
  mkdir -p "$d"
  log "Terminal-Bench oracle check ($TB_DATASET, native build=$TB_NATIVE_BUILD)"
  "$HARBOR_VENV/bin/harbor" run "${HARBOR_ARGS[@]}" --agent oracle --n-concurrent 4 \
    --jobs-dir "$d" --job-name job </dev/null >"$RUN/logs/harbor-oracle.log" 2>&1 \
    || warn "harbor oracle run exited non-zero (logs/harbor-oracle.log)"
  jq -n --arg ds "$TB_DATASET" --argjson nb "$TB_NATIVE_BUILD" --arg arch "$(docker version -f '{{.Server.Arch}}' 2>/dev/null)" \
    '{model:"oracle", scope:"all", dataset:$ds, agent:"oracle", native_build:($nb==1), docker_arch:$arch}' >"$d/meta.json"
}

case "$SUITE" in
  quick)
    BENCH_REPS=3
    suite_perf "0,16384"
    suite_server "0 16384" 0
    suite_agent smoke ;;
  standard|full)
    suite_perf "$BENCH_DEPTHS"
    suite_server "$BENCHY_DEPTHS" 1
    if [[ $SUITE == full ]]; then suite_coding "humaneval mbpp"; else suite_coding "$EVALPLUS_DATASETS"; fi
    suite_agent "$([[ $SUITE == full ]] && echo verified || echo "$SWE_SUBSET")" ;;
  final)   # publishable device results for one model: no SWE-bench, no MLX
    suite_perf "$BENCH_DEPTHS"
    suite_server "$BENCHY_DEPTHS" 1
    suite_coding "$EVALPLUS_DATASETS" ;;
  perf)    suite_perf "$BENCH_DEPTHS" ;;
  ref4167) suite_ref4167 ;;
  server)  suite_server "$BENCHY_DEPTHS" 1 ;;
  mlx)     suite_mlx ;;
  coding)  suite_coding "$EVALPLUS_DATASETS" ;;
  rescore) suite_rescore ;;
  agent)   suite_agent "${2:-$SWE_SUBSET}" ;;
  tbench)  suite_tbench "${2:-all}" ;;
  tbench-check) suite_tbench_check ;;
  *) die "unknown suite: $SUITE (quick|final|standard|full|perf|ref4167|server|mlx|coding|rescore|agent|tbench|tbench-check|collect|compare)" ;;
esac

