#!/usr/bin/env bash
# Run benchmark suites and collect results into results/<run>/.
#
#   ./run.sh quick      # ~30-90 min: llama-bench, llama-benchy, 1 SWE-bench task
#   ./run.sh standard   # overnight+: full depth sweeps, prefix cache, concurrency,
#                       #   EvalPlus, SWE-bench Verified Mini (50)
#   ./run.sh full       # days: standard + SWE-bench Verified (500) + MBPP+
#
#   Individual suites: perf | server | coding | agent
#   Machine-level (no GGUF model, run once): ref4167 | mlx
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

MODEL="${MODEL:-$(model_labels | head -n1)}"
[[ -n "$(model_glob "$MODEL")" ]] || die "Unknown MODEL '$MODEL'. Labels in GGUF_MODELS: $(model_labels | tr '\n' ' ')"
[[ -n "$(model_path "$MODEL")" ]] || die "Model $MODEL not downloaded; run ./bootstrap.sh"
MODELS_SEL="$MODEL"

STAMP="$(date +%Y%m%d-%H%M)"
case "$SUITE" in  # these two don't use the GGUF model
  ref4167|mlx) RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${SUITE}}" ;;
  *)           RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${MODEL_NAME}_${MODEL}_${SUITE}}" ;;
esac
RUN="$(mkdir -p "$RUN" && cd "$RUN" && pwd)"
mkdir -p "$RUN"/{llama_bench,benchy,server,mlx,evalplus,swebench,power,logs}
SERVER_PID="" PROXY_PID="" MLX_PID="" POWER_PID=""

# Always stop servers and write summary.md, also when a step dies part-way.
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_power; stop_pid "$PROXY_PID"; stop_pid "$SERVER_PID"; stop_pid "$MLX_PID"
  python "$ROOT/bench/collect.py" "$RUN" >/dev/null 2>&1 || true
  if [[ -s "$RUN/problems.log" ]]; then
    warn "Run finished with problems (rc=$rc); see $RUN/problems.log and $RUN/logs/"
  fi
  log "Summary: $RUN/summary.md"
  exit "$rc"
}
trap cleanup EXIT INT TERM
caffeinate -dimsu -w $$ &   # keep the Mac awake for the whole run

MTP_ARGS=()
if [[ "$USE_MTP" == 1 ]] && llama_supports_mtp; then
  MTP_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$MTP_DRAFT_N")
elif [[ "$USE_MTP" == 1 ]]; then
  warn "This llama.cpp build has no draft-mtp; all MTP runs are skipped"
fi

if [[ "$SUITE" == ref4167 || "$SUITE" == mlx ]]; then MODELS_SEL=""; fi
models_json="$(for m in $MODELS_SEL; do f="$(model_path "$m")"
  jq -nc --arg l "$m" --arg r "$(model_repo "$m")" --arg f "$(basename "$f")" --arg b "$(stat -f %z "$f")" \
    '{label:$l, repo:$r, file:$f, bytes:($b|tonumber)}'; done | jq -sc .)"
python "$ROOT/bench/sysinfo.py" "$RUN/env.json" "$LLAMA_DIR" "$MODEL_NAME" \
  "$(jq -nc --arg s "$SUITE" --argjson models "$models_json" --arg kv "$KV_CACHE_TYPE" \
      --arg ctx "$SERVER_CTX" --arg mtp "${MTP_ARGS[*]:-}" \
      '{suite:$s, models:$models, kv_cache:$kv, server_ctx:($ctx|tonumber), mtp_args:$mtp}')"
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
  for pat in '*[Ff]16*.gguf' '*Q8_0*.gguf' '*Q4_0*.gguf'; do
    local f; f="$(find "$d" -name "$pat" 2>/dev/null | head -n1)"
    if [[ -n "$f" ]]; then args+=(-m "$f"); fi
  done
  (( ${#args[@]} )) || { warn "Llama-2 7B reference models missing; skipping #4167 run"; return; }
  local delay=(); "$LLAMA_BIN/llama-bench" --help 2>&1 | grep -q -- "--delay" && delay=(--delay 30)
  log "llama-bench Llama-2 7B reference (Discussion #4167)"
  "$LLAMA_BIN/llama-bench" ${args[@]+"${args[@]}"} -p 512 -n 128 -ngl 99 ${delay[@]+"${delay[@]}"} -o md \
    >"$RUN/llama_bench/4167.md" 2>"$RUN/logs/ref4167.log" || warn "reference run failed"
}

# ---- server: llama-benchy (pp/tg at depth, TTFT, concurrency, prefix cache) ----
#      + server_bench.py (short-chat inter-token latency percentiles)
TOKENIZER="$MODELS/tokenizer"

run_benchy() {  # <label> <port> <served-model-name> <tokenizer> <depths> [extra llama-benchy args...]
  local label="$1" port="$2" served="$3" tok="$4" depths="$5"; shift 5
  log "llama-benchy $label (depths $depths)"
  # shellcheck disable=SC2086  # word-split the space-separated lists
  llama-benchy --base-url "http://127.0.0.1:$port/v1" --model "$MODEL_NAME" --served-model-name "$served" \
    --tokenizer "$tok" --book-url "$BENCHY_BOOK_URL" --pp $BENCHY_PP --tg "$BENCHY_TG" --depth $depths --runs "$LATENCY_REPS" \
    --latency-mode generation --exact-tg --extra-body 'chat_template_kwargs={"enable_thinking":false}' \
    --format json --save-result "$RUN/benchy/$label.json" "$@" \
    >"$RUN/logs/benchy-$label.log" 2>&1 || warn "llama-benchy $label failed (logs/benchy-$label.log)"
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
  [[ -d "$TOKENIZER" ]] || die "Tokenizer missing at $TOKENIZER; run ./bootstrap.sh"
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
  local np="$SWE_WORKERS"
  for m in $MODELS_SEL; do
    local d="$RUN/swebench/$m"
    mkdir -p "$d"
    start_quality_server "$m" agent -c $((SERVER_CTX * np)) -np "$np" || { warn "skip SWE-bench for $m"; continue; }
    python "$ROOT/bench/telemetry_proxy.py" --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
      --log "$d/calls.jsonl" & PROXY_PID=$!
    python "$ROOT/bench/swe_config.py" --api-base "http://127.0.0.1:$PROXY_PORT/v1" --model "$MODEL_NAME-$m" \
      --step-limit "$SWE_STEP_LIMIT" --thinking "$AGENT_THINKING" --out "$d/mini-swe-agent.yaml"
    log "mini-swe-agent on SWE-bench [$subset] with $m (mtp=$SERVED_MTP, $np worker(s))"
    local t0=$SECONDS
    MSWEA_COST_TRACKING=ignore_errors mini-extra swebench --subset "$ds" --split test \
      -c "$d/mini-swe-agent.yaml" -w "$np" -o "$d/trajs" ${args[@]+"${args[@]}"} \
      >"$RUN/logs/mini-swe-agent-$m.log" 2>&1 || warn "mini-swe-agent exited non-zero (logs/mini-swe-agent-$m.log)"
    local wall=$((SECONDS - t0))
    stop_pid "$PROXY_PID"; PROXY_PID=""; stop_pid "$SERVER_PID"; SERVER_PID=""
    jq -n --arg s "$subset" --arg ds "$ds" --argjson sl "$SWE_STEP_LIMIT" --argjson th "$AGENT_THINKING" \
       --argjson w "$wall" --argjson np "$np" --argjson mtp "$SERVED_MTP" \
       --arg v "$(pip show mini-swe-agent 2>/dev/null | awk '/^Version/{print $2}')" \
       '{subset:$s, dataset:$ds, step_limit:$sl, thinking:($th==1), mtp:($mtp==1), wall_s:$w, workers:$np, mini_swe_agent:$v}' \
       >"$d/meta.json"
    if [[ -f "$d/trajs/preds.json" ]]; then
      cp "$d/trajs/preds.json" "$d/preds.json"
      swe_evaluate "$d" "$m"
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
  perf)    suite_perf "$BENCH_DEPTHS" ;;
  ref4167) suite_ref4167 ;;
  server)  suite_server "$BENCHY_DEPTHS" 1 ;;
  mlx)     suite_mlx ;;
  coding)  suite_coding "$EVALPLUS_DATASETS" ;;
  agent)   suite_agent "${2:-$SWE_SUBSET}" ;;
  *) die "unknown suite: $SUITE (quick|standard|full|perf|ref4167|server|mlx|coding|agent|collect|compare)" ;;
esac

