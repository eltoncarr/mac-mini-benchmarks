#!/usr/bin/env bash
# Run benchmark suites and collect results into results/<run>/.
#
#   ./run.sh quick      # ~30-60 min: llama-bench, server latency, 1 SWE-bench task
#   ./run.sh standard   # overnight: + all quants, depth sweep, MTP on/off, MLX,
#                       #   #4167 reference, EvalPlus, SWE-bench Verified Mini (50)
#   ./run.sh full       # days: standard + SWE-bench Verified (500) + MBPP+
#
#   Individual suites: perf | ref4167 | server | mlx | coding | agent
#   ./run.sh collect results/<run>      # rebuild summary.md
#
# RUN_DIR=results/<run> ./run.sh agent  adds a suite to an existing run.
source "$(dirname "$0")/scripts/lib.sh"
activate_venv

SUITE="${1:-quick}"
if [[ "$SUITE" == collect ]]; then python "$ROOT/bench/collect.py" "${2:?run dir}"; exit; fi

STAMP="$(date +%Y%m%d-%H%M)"
RUN="${RUN_DIR:-$ROOT/results/${STAMP}_${MODEL_NAME}_${SUITE}}"
RUN="$(mkdir -p "$RUN" && cd "$RUN" && pwd)"
mkdir -p "$RUN"/{llama_bench,server,mlx,evalplus,swebench,power,logs}
PRIMARY="$(primary_quant)"
PRIMARY_GGUF="$(gguf_path "$PRIMARY")"
[[ -n "$PRIMARY_GGUF" ]] || die "Primary quant $PRIMARY not downloaded; run ./bootstrap.sh"
CORPUS="$LLAMA_DIR/src"
SERVER_PID="" PROXY_PID="" MLX_PID="" POWER_PID=""

cleanup() { stop_power; stop_pid "$PROXY_PID"; stop_pid "$SERVER_PID"; stop_pid "$MLX_PID"; }
trap cleanup EXIT INT TERM
caffeinate -dimsu -w $$ &   # keep the Mac awake for the whole run

MTP_ARGS=()
if [[ "$USE_MTP" == 1 ]] && llama_supports_mtp; then
  MTP_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$MTP_DRAFT_N")
fi

python "$ROOT/bench/sysinfo.py" "$RUN/env.json" "$LLAMA_DIR" "$MODEL_NAME" \
  "$(jq -nc --arg s "$SUITE" --arg q "$GGUF_QUANTS" --arg r "$GGUF_REPO" --arg kv "$KV_CACHE_TYPE" \
      --arg ctx "$SERVER_CTX" --arg mtp "${MTP_ARGS[*]:-}" \
      '{suite:$s, gguf_repo:$r, quants:($q|split(" ")), kv_cache:$kv, server_ctx:($ctx|tonumber), mtp_args:$mtp}')"
log "Run directory: $RUN"

# ---- perf: llama-bench pp512/tg128 + context-depth sweep ------------------------
suite_perf() {
  local quants="$1" depths="$2"
  for q in $quants; do
    local g; g="$(gguf_path "$q")"
    [[ -n "$g" ]] || { warn "skip $q (not downloaded)"; continue; }
    log "llama-bench $q (depths $depths)"
    start_power "$RUN/power/llama-bench-$q.txt"
    "$LLAMA_BIN/llama-bench" -m "$g" -p 512 -n 128 -d "$depths" -ngl 99 -fa 1 \
      -r "$BENCH_REPS" -o json >"$RUN/llama_bench/$q.json" 2>"$RUN/logs/llama-bench-$q.log" \
      || warn "llama-bench $q failed (see logs/llama-bench-$q.log)"
    stop_power
  done
}

# ---- ref4167: exact command from llama.cpp Discussion #4167 ----------------------
suite_ref4167() {
  local d="$MODELS/llama2-7b" args=()
  for pat in '*[Ff]16*.gguf' '*Q8_0*.gguf' '*Q4_0*.gguf'; do
    local f; f="$(find "$d" -name "$pat" 2>/dev/null | head -n1)"
    [[ -n "$f" ]] && args+=(-m "$f")
  done
  (( ${#args[@]} )) || { warn "Llama-2 7B reference models missing; skipping #4167 run"; return; }
  local delay=(); "$LLAMA_BIN/llama-bench" --help 2>&1 | grep -q -- "--delay" && delay=(--delay 30)
  log "llama-bench Llama-2 7B reference (Discussion #4167)"
  "$LLAMA_BIN/llama-bench" ${args[@]+"${args[@]}"} -p 512 -n 128 -ngl 99 ${delay[@]+"${delay[@]}"} -o md \
    >"$RUN/llama_bench/4167.md" 2>"$RUN/logs/ref4167.log" || warn "reference run failed"
}

# ---- server: TTFT / TPOT / ITL / prefill at depth / cache / concurrency ----------
run_server_bench() {  # <label> <scenarios> [extra server_bench args...]
  local label="$1" scen="$2"; shift 2
  python "$ROOT/bench/server_bench.py" --engine llama --model "$MODEL_NAME" \
    --base-url "http://127.0.0.1:$SERVER_PORT/v1" --label "$label" --scenarios "$scen" \
    --reps "$LATENCY_REPS" --sizes "$LONGCTX_SIZES" --concurrency "$CONCURRENCY_LEVELS" \
    --corpus "$CORPUS" --no-think --out "$RUN/server/$label.json" "$@" \
    || warn "server bench $label failed"
}

suite_server() {
  local quants="$1" full="$2"
  for q in $quants; do
    local g; g="$(gguf_path "$q")"; [[ -n "$g" ]] || continue
    start_llama_server "$g" "$RUN/logs/server-$q.log" -c "$SERVER_CTX" -np 1
    start_power "$RUN/power/server-$q.txt"
    run_server_bench "llama.cpp-$q" "short,longctx$([[ $full == 1 && $q == "$PRIMARY" ]] && echo ,cache)"
    stop_power; stop_pid "$SERVER_PID"; SERVER_PID=""
  done
  if (( ${#MTP_ARGS[@]} )); then
    start_llama_server "$PRIMARY_GGUF" "$RUN/logs/server-$PRIMARY-mtp.log" -c "$SERVER_CTX" -np 1 ${MTP_ARGS[@]+"${MTP_ARGS[@]}"}
    start_power "$RUN/power/server-$PRIMARY-mtp.txt"
    run_server_bench "llama.cpp-$PRIMARY-mtp" "short$([[ $full == 1 ]] && echo ,longctx)"
    stop_power; stop_pid "$SERVER_PID"; SERVER_PID=""
  fi
  if [[ $full == 1 ]]; then
    local np=4
    start_llama_server "$PRIMARY_GGUF" "$RUN/logs/server-$PRIMARY-np$np.log" -c $((np * 16384)) -np "$np"
    run_server_bench "llama.cpp-$PRIMARY-np$np" "concurrency"
    stop_pid "$SERVER_PID"; SERVER_PID=""
  fi
}

# ---- mlx: mlx_lm.benchmark + mlx_lm.server latency --------------------------------
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
  if wait_for_http "http://127.0.0.1:$MLX_PORT/v1/models" 600; then
    python "$ROOT/bench/server_bench.py" --engine mlx --model "$m" --base-url "http://127.0.0.1:$MLX_PORT/v1" \
      --label "mlx-4bit" --scenarios short,longctx --reps "$LATENCY_REPS" --sizes "1024 4096 16384 32768" \
      --corpus "$CORPUS" --no-think --out "$RUN/server/mlx-4bit.json" || warn "mlx server bench failed"
  else
    warn "mlx_lm.server did not start (logs/mlx-server.log)"
  fi
  stop_pid "$MLX_PID"; MLX_PID=""
}

# ---- coding: EvalPlus HumanEval+/MBPP+ --------------------------------------------
suite_coding() {
  local datasets="$1"
  start_llama_server "$PRIMARY_GGUF" "$RUN/logs/server-coding.log" -c 32768 -np 1 ${MTP_ARGS[@]+"${MTP_ARGS[@]}"} \
    --chat-template-kwargs "{\"enable_thinking\": $([[ $EVALPLUS_THINKING == 1 ]] && echo true || echo false)}"
  for ds in $datasets; do
    log "EvalPlus $ds"
    OPENAI_API_KEY=local evalplus.evaluate --model "$MODEL_NAME" --dataset "$ds" --backend openai \
      --base-url "http://127.0.0.1:$SERVER_PORT/v1" --greedy --root "$RUN/evalplus/raw" \
      2>&1 | tee "$RUN/evalplus/$ds.log" | tail -n 8 || warn "EvalPlus $ds failed"
  done
  stop_pid "$SERVER_PID"; SERVER_PID=""
}

# ---- agent: SWE-bench with mini-swe-agent (real repos, fix-a-bug tasks) ----------
suite_agent() {
  local subset="$1" ds args=()
  docker info >/dev/null 2>&1 || die "Docker is not running (needed for SWE-bench). Start Docker/Colima."
  case "$subset" in
    smoke)    ds=verified; args=(--filter "^${SWE_SMOKE_INSTANCE}\$") ;;
    mini)     ds=MariusHobbhahn/swe-bench-verified-mini ;;
    verified) ds=verified ;;
    *) die "unknown SWE subset $subset" ;;
  esac
  local d="$RUN/swebench" np="$SWE_WORKERS"
  start_llama_server "$PRIMARY_GGUF" "$RUN/logs/server-agent.log" -c $((SERVER_CTX * np)) -np "$np" ${MTP_ARGS[@]+"${MTP_ARGS[@]}"}
  python "$ROOT/bench/telemetry_proxy.py" --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
    --log "$d/calls.jsonl" & PROXY_PID=$!
  python "$ROOT/bench/swe_config.py" --api-base "http://127.0.0.1:$PROXY_PORT/v1" --model "$MODEL_NAME" \
    --step-limit "$SWE_STEP_LIMIT" --thinking "$AGENT_THINKING" --out "$d/mini-swe-agent.yaml"
  log "mini-swe-agent on SWE-bench [$subset] ($np worker(s)); trajectories in $d/trajs"
  local t0=$SECONDS
  MSWEA_COST_TRACKING=ignore_errors mini-extra swebench --subset "$ds" --split test \
    -c "$d/mini-swe-agent.yaml" -w "$np" -o "$d/trajs" ${args[@]+"${args[@]}"} \
    >"$RUN/logs/mini-swe-agent.log" 2>&1 || warn "mini-swe-agent exited non-zero (logs/mini-swe-agent.log)"
  local wall=$((SECONDS - t0))
  stop_pid "$PROXY_PID"; PROXY_PID=""; stop_pid "$SERVER_PID"; SERVER_PID=""
  jq -n --arg s "$subset" --arg ds "$ds" --argjson sl "$SWE_STEP_LIMIT" --argjson th "$AGENT_THINKING" \
     --argjson w "$wall" --argjson np "$np" --arg v "$(pip show mini-swe-agent 2>/dev/null | awk '/^Version/{print $2}')" \
     '{subset:$s, dataset:$ds, step_limit:$sl, thinking:($th==1), wall_s:$w, workers:$np, mini_swe_agent:$v}' >"$d/meta.json"
  [[ -f "$d/trajs/preds.json" ]] || { warn "no preds.json produced"; return; }
  cp "$d/trajs/preds.json" "$d/preds.json"
  swe_evaluate "$d"
}

swe_evaluate() {
  local d="$1" run_id; run_id="$(basename "$RUN" | tr -c 'A-Za-z0-9_.\n-' '_')"
  case "$SWE_EVAL" in
    none) log "SWE_EVAL=none: predictions saved, not evaluated" ;;
    sbcli)
      [[ -n "${SWEBENCH_API_KEY:-}" ]] || { warn "SWEBENCH_API_KEY not set; skipping sb-cli"; return; }
      log "Submitting to SWE-bench cloud evaluator (sb-cli)"
      mkdir -p "$d/sbcli"
      sb-cli submit swe-bench_verified test --predictions_path "$d/preds.json" --run_id "$run_id" \
        --output_dir "$d/sbcli" >"$RUN/logs/sbcli.log" 2>&1 || warn "sb-cli failed (logs/sbcli.log)"
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
        >"$RUN/logs/swebench-eval.log" 2>&1 || warn "SWE-bench evaluation failed (logs/swebench-eval.log)"
      local r; r="$(ls "$d"/*."$run_id".json 2>/dev/null | head -n1)"
      if [[ -n "$r" ]]; then mv "$r" "$d/report.json"; fi ;;
  esac
}

case "$SUITE" in
  quick)
    BENCH_REPS=3
    suite_perf "$PRIMARY" "0,16384"
    suite_server "$PRIMARY" 0
    suite_agent smoke ;;
  standard|full)
    suite_perf "$GGUF_QUANTS" "$BENCH_DEPTHS"
    if [[ "$RUN_4167_REFERENCE" == 1 ]]; then suite_ref4167; fi
    suite_server "$GGUF_QUANTS" 1
    suite_mlx
    if [[ $SUITE == full ]]; then suite_coding "humaneval mbpp"; else suite_coding "$EVALPLUS_DATASETS"; fi
    suite_agent "$([[ $SUITE == full ]] && echo verified || echo "$SWE_SUBSET")" ;;
  perf)    suite_perf "$GGUF_QUANTS" "$BENCH_DEPTHS" ;;
  ref4167) suite_ref4167 ;;
  server)  suite_server "$GGUF_QUANTS" 1 ;;
  mlx)     suite_mlx ;;
  coding)  suite_coding "$EVALPLUS_DATASETS" ;;
  agent)   suite_agent "${2:-$SWE_SUBSET}" ;;
  *) die "unknown suite: $SUITE (quick|standard|full|perf|ref4167|server|mlx|coding|agent|collect)" ;;
esac

python "$ROOT/bench/collect.py" "$RUN"
log "Done. Summary: $RUN/summary.md"
