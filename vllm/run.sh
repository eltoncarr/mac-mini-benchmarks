#!/usr/bin/env bash
# vLLM (vllm-metal) counterpart of ./run.sh: same suites, same clients, same results
# layout, so ./run.sh compare can put llama.cpp and vLLM runs side by side.
#
#   vllm/run.sh quick      # ~30-90 min: vllm bench serve sweep, llama-benchy, 1 SWE-bench task
#   vllm/run.sh final      # publishable set: full depth sweep, server suite, HumanEval+
#   vllm/run.sh standard   # overnight+: final + SWE-bench Verified Mini (50)
#   vllm/run.sh full       # days: standard + SWE-bench Verified (500) + MBPP+
#
#   Individual suites: perf | server | coding | agent [smoke|mini|verified]
#   RUN_DIR=results/<run> vllm/run.sh coding    # add a suite to an existing run
#   ./run.sh collect results/<run> / ./run.sh compare results/<run> ...   (shared)
#
# One pass benchmarks one model: the first in VLLM_MODELS (vllm/config.env), or
#   MODEL=mlx-8bit vllm/run.sh quick
source "$(dirname "$0")/lib.sh"
activate_venv
[[ -x "$VLLM_BIN/vllm" ]] || die "No vLLM at $VLLM_VENV. Run vllm/bootstrap.sh first."

SUITE="${1:-quick}"
MODEL="${MODEL:-$(vllm_labels | head -n1)}"
[[ -n "$(vllm_repo "$MODEL")" ]] || die "Unknown MODEL '$MODEL'. Labels in VLLM_MODELS: $(vllm_labels | tr '\n' ' ')"
vllm_model_ok "$MODEL" || die "Model $MODEL not downloaded; run vllm/bootstrap.sh"
MDIR="$(vllm_model_dir "$MODEL")"
LABEL="vllm-$MODEL"   # prefix of every result file, like llama.cpp-<model>

STAMP="$(date +%Y%m%d-%H%M)"
RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${MODEL_NAME}_vllm-${MODEL}_${SUITE}}"
RUN="$(mkdir -p "$RUN" && cd "$RUN" && pwd)"
mkdir -p "$RUN"/{vllm_bench,benchy,server,evalplus,swebench,power,logs}
SERVER_PID="" PROXY_PID="" POWER_PID=""

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if (( rc != 0 )); then
    echo "[$(date +%H:%M:%S)] error: run stopped early (exit $rc; Ctrl-C or a fatal error); later suites did not run" >>"$RUN/problems.log"
  fi
  stop_power; stop_pid "$PROXY_PID"; stop_server
  python "$ROOT/bench/collect.py" "$RUN" >/dev/null 2>&1 || true
  if [[ -s "$RUN/problems.log" ]]; then
    warn "Run finished with problems (rc=$rc); see $RUN/problems.log and $RUN/logs/"
  fi
  log "Summary: $RUN/summary.md"
  exit "$rc"
}
trap cleanup EXIT INT TERM
caffeinate -dimsu -w $$ &
if [[ "${MEASURE_POWER:-0}" == 1 ]]; then
  log "MEASURE_POWER=1: powermetrics needs sudo for the whole run"
  sudo -v || die "sudo is required for MEASURE_POWER=1"
  while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 60; done &
fi

# Flags every server gets. Prefix caching is off unless a step asks for it, as in
# vllm-metal's benchmarking guide for uncached comparisons.
BASE_ARGS=(--host 127.0.0.1 --port "$VLLM_PORT" --served-model-name "$MODEL_NAME" "$MODEL_NAME-$MODEL"
  --max-model-len "$VLLM_MAX_MODEL_LEN" --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
  --max-num-batched-tokens "$VLLM_MAX_BATCHED_TOKENS" --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE"
  --enable-prompt-tokens-details)
[[ -n "$VLLM_REASONING_PARSER" ]] && BASE_ARGS+=(--reasoning-parser "$VLLM_REASONING_PARSER")
NGRAM_ARGS=(--speculative-config "$NGRAM_SPEC_CONFIG" --no-async-scheduling --mamba-ssm-cache-dtype float32)

python "$ROOT/bench/sysinfo.py" "$RUN/env.json" "$VENDOR/none" "$MODEL_NAME" \
  "$(jq -nc --arg s "$SUITE" --arg l "$MODEL" --arg r "$(vllm_repo "$MODEL")" \
      --arg b "$(du -sk "$MDIR" | awk '{print $1 * 1024}')" --arg sl "$LABEL" \
      --argjson v "$(vllm_versions)" --arg args "${BASE_ARGS[*]}" --arg ng "${NGRAM_ARGS[*]}" \
      --arg ctx "$VLLM_MAX_MODEL_LEN" --arg kv "$VLLM_KV_CACHE_DTYPE" --arg pp "$VLLM_PP" \
      '{suite:$s, engine:("vllm-metal " + ($v["vllm-metal"] // "?")), engine_versions:$v,
        models:[{label:$l, repo:$r, file:"(MLX safetensors)", bytes:($b|tonumber), server_label:$sl, spec:"ngram"}],
        kv_cache:$kv, server_ctx:($ctx|tonumber), perf_pp:($pp|tonumber), server_args:$args, spec_args:$ng}')"
log "Run directory: $RUN"

# start_server <tag> [extra vllm serve args...]; sets SERVER_PID, returns 1 on failure.
# CHAT_KWARGS=... before the call replaces the default chat-template kwargs.
start_server() {
  local tag="$1"; shift
  local logf="$RUN/logs/vllm-$MODEL-$tag.log"
  local kw="${CHAT_KWARGS:-$VLLM_CHAT_KWARGS}" kwargs=()
  [[ -n "$kw" && "$kw" != "{}" ]] && kwargs=(--default-chat-template-kwargs "$kw")
  VLLM_ENABLE_V1_MULTIPROCESSING=1 "$VLLM_BIN/vllm" serve "$MDIR" "${BASE_ARGS[@]}" \
    ${kwargs[@]+"${kwargs[@]}"} "$@" >"$logf" 2>&1 &
  SERVER_PID=$!
  if ! wait_for_http "http://127.0.0.1:$VLLM_PORT/v1/models" 1200; then
    tail -n 30 "$logf" >&2
    warn "vllm serve ($tag) did not become ready (log: $logf)"
    stop_server
    return 1
  fi
  log "vllm serve up (pid $SERVER_PID, $tag): $*"
}

# vllm serve runs the engine in child processes; stop the whole group.
stop_server() {
  [[ -n "${SERVER_PID:-}" ]] || return 0
  pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
  stop_pid "$SERVER_PID"; SERVER_PID=""
  local t=0
  while curl -sf --noproxy '*' "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 && (( t < 30 )); do sleep 1; t=$((t + 1)); done
}

# Prometheus snapshot after a phase: spec-decode draft/accept counters, prefix-cache hits.
save_metrics() {  # <name>
  curl -sf --noproxy '*' "http://127.0.0.1:$VLLM_PORT/metrics" >"$RUN/server/$1.metrics.txt" 2>/dev/null || true
}

# True once the server has drafted n-gram tokens (greedy request with repeated spans).
spec_active() {
  curl -sf --noproxy '*' "http://127.0.0.1:$VLLM_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL_NAME" '{model:$m, temperature:0, max_tokens:96,
          messages:[{role:"user", content:"Repeat exactly, 8 times, one per line: the quick brown fox jumps over the lazy dog"}]}')" \
    >/dev/null || return 1
  local metrics n
  metrics="$(curl -sf --noproxy '*' "http://127.0.0.1:$VLLM_PORT/metrics")" || return 1
  if ! grep -q '^vllm:spec_decode_num_draft_tokens' <<<"$metrics"; then
    warn "no spec-decode counters in /metrics; cannot confirm n-gram drafting, measuring anyway"
    return 0
  fi
  n="$(awk '/^vllm:spec_decode_num_draft_tokens/{s += $NF} END{print int(s)}' <<<"$metrics")"
  [[ "${n:-0}" -gt 0 ]]
}

# vllm bench serve, random prompts. <name> <input len> <output len> <concurrency> <prompts>
vllm_bench() {
  local name="$1" in="$2" out="$3" c="$4" n="$5"
  log "vllm bench serve $name: in $in, out $out, concurrency $c, $n prompts"
  "$VLLM_BIN/vllm" bench serve --backend openai --base-url "http://127.0.0.1:$VLLM_PORT" --endpoint /v1/completions \
    --model "$MDIR" --served-model-name "$MODEL_NAME" \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" --random-range-ratio 0 \
    --num-prompts "$n" --num-warmups "$(( in > 8192 ? 1 : 2 ))" --request-rate inf --max-concurrency "$c" \
    --ignore-eos --temperature 0 --seed 713 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$RUN/vllm_bench" --result-filename "${name}_in${in}_out${out}_c${c}.json" \
    >>"$RUN/logs/vllm-bench-$name.log" 2>&1 || warn "vllm bench serve $name (in $in, c$c) failed (logs/vllm-bench-$name.log)"
}

# ---- perf: single-request sweep over context depth (llama-bench analogue) ----------
#      prompt = depth + VLLM_PP tokens, VLLM_TG output tokens; prefill t/s = prompt/TTFT,
#      decode t/s = 1000/TPOT. Through the server, so not engine-only like llama-bench.
suite_perf() {  # <comma-separated depths>
  local depths="$1" d
  start_server perf --max-num-seqs 1 --no-enable-prefix-caching || { warn "skip perf"; return; }
  start_power "$RUN/power/vllm-bench-$MODEL.txt"
  for d in ${depths//,/ }; do
    vllm_bench "$LABEL-perf" $((d + VLLM_PP)) "$VLLM_TG" 1 "$BENCH_REPS"
  done
  stop_power; stop_server
}

# ---- server: llama-benchy + short-chat latency, plain and n-gram; prefix cache;
#      concurrency (llama-benchy levels for cross-engine tables, vllm bench serve sweep)
run_benchy() {  # <label> <depths> [extra llama-benchy args...]
  local label="$1" depths="$2"; shift 2
  log "llama-benchy $label (depths $depths)"
  # shellcheck disable=SC2086
  llama-benchy --base-url "http://127.0.0.1:$VLLM_PORT/v1" --model "$MODEL_NAME" --served-model-name "$MODEL_NAME" \
    --tokenizer "$MDIR" --book-url "$BENCHY_BOOK_URL" --pp $BENCHY_PP --tg "$BENCHY_TG" --depth $depths --runs "$LATENCY_REPS" \
    --latency-mode generation --exact-tg --extra-body 'chat_template_kwargs={"enable_thinking":false}' \
    --format json --save-result "$RUN/benchy/$label.json" "$@" \
    >>"$RUN/logs/benchy-$label.log" 2>&1 || warn "llama-benchy $label failed (logs/benchy-$label.log)"
}

run_latency() {  # <label>; greedy so the n-gram arm drafts
  python "$ROOT/bench/server_bench.py" --base-url "http://127.0.0.1:$VLLM_PORT/v1" --model "$MODEL_NAME" --label "$1" \
    --reps "$LATENCY_REPS" --temperature 0 --out "$RUN/server/$1.json" || warn "server latency $1 failed"
}

suite_server() {  # <depths> <full: 1 adds prefix-cache + concurrency runs>
  local depths="$1" full="$2"
  ensure_benchy_book || warn "llama-benchy runs will likely fail (no prompt text)"
  # Server-side default temperature 0 for clients that send none, in both arms, so
  # plain vs n-gram compares the same decoding (n-gram drafts greedy requests only).
  local greedy=(--override-generation-config '{"temperature": 0}')
  if start_server plain --max-num-seqs 1 --no-enable-prefix-caching "${greedy[@]}"; then
    start_power "$RUN/power/server-$MODEL.txt"
    run_benchy "$LABEL" "$depths" --no-cache
    run_latency "$LABEL"
    stop_power; save_metrics "$LABEL"; stop_server
  fi
  if [[ "$USE_NGRAM" == 1 ]] && start_server ngram --max-num-seqs 1 --no-enable-prefix-caching "${greedy[@]}" "${NGRAM_ARGS[@]}"; then
    if spec_active; then
      start_power "$RUN/power/server-$MODEL-ngram.txt"
      run_benchy "$LABEL-ngram" "$depths" --no-cache
      run_latency "$LABEL-ngram"
      stop_power; save_metrics "$LABEL-ngram"
    else
      warn "$MODEL: n-gram speculation configured but no drafts were made; n-gram runs skipped"
    fi
    stop_server
  fi
  [[ $full == 1 ]] || return 0
  if start_server prefix --max-num-seqs 1 --enable-prefix-caching "${greedy[@]}"; then
    run_benchy "$LABEL-prefix-cache" "4096 16384 32768" --enable-prefix-caching
    save_metrics "$LABEL-prefix-cache"; stop_server
  fi
  local np c
  np="$(echo "$CONCURRENCY_LEVELS $VLLM_CONCURRENCY_LEVELS" | tr ' ' '\n' | sort -n | tail -n1)"
  if start_server "np$np" --max-num-seqs "$np" --no-enable-prefix-caching "${greedy[@]}"; then
    # shellcheck disable=SC2086
    run_benchy "$LABEL-concurrency" "0 4096" --no-cache --concurrency $CONCURRENCY_LEVELS
    for c in $VLLM_CONCURRENCY_LEVELS; do
      vllm_bench "$LABEL-sweep" "$VLLM_SWEEP_IN" "$VLLM_SWEEP_OUT" "$c" $(( c * 4 > 8 ? c * 4 : 8 ))
    done
    stop_server
  fi
}

# ---- coding: EvalPlus HumanEval+/MBPP+ (greedy) -------------------------------------
export EVALPLUS_MAX_MEMORY_BYTES=-1   # see ./run.sh

suite_coding() {  # <datasets>
  local datasets="$1" ds
  # Plain decoding: n-gram verification on hybrid models re-scans GDN state, and the
  # llama.cpp quality runs (MTP) and these should not differ in more than the engine.
  CHAT_KWARGS="{\"enable_thinking\": $([[ $EVALPLUS_THINKING == 1 ]] && echo true || echo false)}" \
    start_server coding --max-num-seqs 1 --max-model-len 32768 || { warn "skip EvalPlus"; return; }
  mkdir -p "$RUN/evalplus/$MODEL"
  for ds in $datasets; do
    log "EvalPlus $ds on vLLM $MODEL"
    OPENAI_API_KEY=local evalplus.evaluate --model "$MODEL_NAME-$MODEL" --dataset "$ds" --backend openai \
      --base-url "http://127.0.0.1:$VLLM_PORT/v1" --greedy --root "$RUN/evalplus/raw/$MODEL" \
      2>&1 | tee "$RUN/evalplus/$MODEL/$ds.log" | tail -n 8 || warn "EvalPlus $ds failed"
  done
  stop_server
}

# ---- agent: SWE-bench with mini-swe-agent ------------------------------------------
suite_agent() {  # <smoke|mini|verified>
  local subset="$1" ds args=()
  docker info >/dev/null 2>&1 || die "Docker is not running (needed for SWE-bench). Run ./bootstrap.sh or start Colima."
  docker_amd64_ok || die "SWE-bench skipped: Docker cannot run the x86_64 SWE-bench images (see README)"
  case "$subset" in
    smoke)    ds=verified; args=(--filter "^${SWE_SMOKE_INSTANCE}\$") ;;
    mini)     ds=MariusHobbhahn/swe-bench-verified-mini ;;
    verified) ds=verified ;;
    *) die "unknown SWE subset $subset" ;;
  esac
  if [[ -n "$SWE_SLICE" && "$subset" != smoke ]]; then args+=(--slice "$SWE_SLICE"); fi
  local np="$SWE_WORKERS" tag="$subset${SWE_SLICE:+_${SWE_SLICE/:/-}}"
  [[ "$subset" == smoke ]] && tag=smoke
  local d="$RUN/swebench/$MODEL/$tag"
  mkdir -p "$d"
  # The agent sends its own sampling and thinking switch; prefix caching on, as llama-server.
  CHAT_KWARGS="{}" start_server agent --max-num-seqs "$np" --enable-prefix-caching || { warn "skip SWE-bench"; return; }
  python "$ROOT/bench/telemetry_proxy.py" --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$VLLM_PORT" \
    --log "$d/calls.jsonl" & PROXY_PID=$!
  python "$ROOT/bench/swe_config.py" --api-base "http://127.0.0.1:$PROXY_PORT/v1" --model "$MODEL_NAME-$MODEL" \
    --step-limit "$SWE_STEP_LIMIT" --thinking "$AGENT_THINKING" \
    --max-tokens "$AGENT_MAX_TOKENS" --timeout "$AGENT_TIMEOUT_S" --out "$d/mini-swe-agent.yaml"
  log "mini-swe-agent on SWE-bench [$subset] with vLLM $MODEL ($np worker(s))"
  local t0=$SECONDS
  MSWEA_COST_TRACKING=ignore_errors mini-extra swebench --subset "$ds" --split test \
    -c "$d/mini-swe-agent.yaml" -w "$np" -o "$d/trajs" ${args[@]+"${args[@]}"} \
    >"$RUN/logs/mini-swe-agent-$MODEL-$tag.log" 2>&1 || warn "mini-swe-agent exited non-zero (logs/mini-swe-agent-$MODEL-$tag.log)"
  local wall=$((SECONDS - t0))
  save_metrics "$LABEL-agent-$tag"
  stop_pid "$PROXY_PID"; PROXY_PID=""; stop_server
  jq -n --arg model "$MODEL" --arg s "$subset${SWE_SLICE:+[$SWE_SLICE]}" --arg ds "$ds" --argjson sl "$SWE_STEP_LIMIT" \
     --argjson th "$AGENT_THINKING" --argjson w "$wall" --argjson np "$np" \
     --arg v "$(python -c 'import importlib.metadata as m; print(m.version("mini-swe-agent"))' 2>/dev/null)" \
     '{model:$model, engine:"vllm", subset:$s, dataset:$ds, step_limit:$sl, thinking:($th==1), mtp:false,
       wall_s:$w, workers:$np, mini_swe_agent:$v}' >"$d/meta.json"
  if [[ -f "$d/trajs/preds.json" ]]; then
    cp "$d/trajs/preds.json" "$d/preds.json"
    swe_evaluate "$d" "$MODEL-$tag"
  else
    warn "no preds.json produced for $MODEL"
  fi
}

# Same evaluation as ./run.sh (local harness in Docker, or sb-cli).
swe_evaluate() {
  local d="$1" run_id; run_id="$(basename "$RUN" | tr -c 'A-Za-z0-9_.\n-' '_')_$2"
  case "$SWE_EVAL" in
    none) log "SWE_EVAL=none: predictions saved, not evaluated" ;;
    sbcli)
      [[ -n "${SWEBENCH_API_KEY:-}" ]] || { warn "SWEBENCH_API_KEY not set; skipping sb-cli"; return; }
      mkdir -p "$d/sbcli"
      sb-cli submit swe-bench_verified test --predictions_path "$d/preds.json" --run_id "$run_id" \
        --output_dir "$d/sbcli" >"$RUN/logs/sbcli-$2.log" 2>&1 || warn "sb-cli failed (logs/sbcli-$2.log)"
      local r; r="$(grep -l resolved_instances "$d"/sbcli/*.json 2>/dev/null | head -n1)"
      if [[ -n "$r" ]]; then cp "$r" "$d/report.json"; fi ;;
    local)
      log "Evaluating patches with the SWE-bench harness (Docker)"
      local ids; ids=$(jq -r 'keys[]' "$d/preds.json" | tr '\n' ' ')
      # shellcheck disable=SC2086
      (cd "$d" && python -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Verified --split test \
        --predictions_path "$d/preds.json" --instance_ids $ids \
        --max_workers 4 --run_id "$run_id" --report_dir "$d") \
        >"$RUN/logs/swebench-eval-$2.log" 2>&1 || warn "SWE-bench evaluation failed (logs/swebench-eval-$2.log)"
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
  final)
    suite_perf "$BENCH_DEPTHS"
    suite_server "$BENCHY_DEPTHS" 1
    suite_coding "$EVALPLUS_DATASETS" ;;
  perf)    suite_perf "$BENCH_DEPTHS" ;;
  server)  suite_server "$BENCHY_DEPTHS" 1 ;;
  coding)  suite_coding "$EVALPLUS_DATASETS" ;;
  agent)   suite_agent "${2:-$SWE_SUBSET}" ;;
  *) die "unknown suite: $SUITE (quick|final|standard|full|perf|server|coding|agent)" ;;
esac
