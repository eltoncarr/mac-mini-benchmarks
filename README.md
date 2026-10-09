# mac-mini-benchmarks

Reproducible local-LLM benchmarks for an **Apple Mac mini M5 Pro (64 GB)**, focused on
**Qwen3.8-27B** (dense, Apache 2.0). It covers raw inference speed, serving latency,
standard coding scores and agentic bug fixing on real repositories, all with
standard tools so the numbers can be posted where the community compares them
(see [PUBLISHING.md](PUBLISHING.md)).

```bash
git clone https://github.com/eltoncarr/mac-mini-benchmarks && cd mac-mini-benchmarks
./bootstrap.sh            # one-time setup: tools, llama.cpp (Metal), Python env, models, Docker
./run.sh quick            # about an hour: speed + latency + one SWE-bench bug fix
./run.sh standard         # overnight or longer: everything below
cat results/*/summary.md
```

## What gets measured

| Suite | Tool (standard) | Metrics |
|---|---|---|
| `perf` | `llama-bench` | **pp512**, **tg128** (t/s ± stddev), plus pp/tg at context depth 4k/16k/32k/64k (`-d`) for each quant |
| `ref4167` | `llama-bench` on Llama-2 7B F16/Q8_0/Q4_0 | the exact command the [llama.cpp Apple Silicon thread](https://github.com/ggml-org/llama.cpp/discussions/4167) uses, so the chip can be compared with every other Mac |
| `server` | `llama-server` + `bench/server_bench.py` | **TTFT**, **TPOT**, inter-token latency p50/p90/p99, decode t/s, e2e latency, effective prefill t/s at 1k–120k tokens of real source code, prompt-cache cold vs warm TTFT, aggregate throughput at 1/2/4 parallel requests; with and without **MTP** speculative decoding |
| `mlx` | `mlx_lm.benchmark`, `mlx_lm.server` | prompt/generation t/s and peak memory for the MLX 4-bit build, plus the same latency suite |
| `coding` | [EvalPlus](https://github.com/evalplus/evalplus) | HumanEval+ (and MBPP+ in `full`) pass@1, greedy |
| `agent` | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on [SWE-bench Verified](https://www.swebench.com/) | % resolved (official harness), plus a **per-call telemetry log** of the agent's real requests: context size, TTFT, prefill/decode t/s, prefix-cache hits and MTP acceptance as the context grows |
| power (optional) | `powermetrics` | average/max CPU+GPU+ANE watts per phase (`MEASURE_POWER=1`) |

Every run also writes `env.json`: chip, GPU cores, RAM, macOS version, GPU wired-memory
limit, llama.cpp commit and Python package versions.

### Agentic coding tasks (fix a bug in a real repo)

SWE-bench tasks are real GitHub issues from real Python projects (Django, SymPy,
Astropy, scikit-learn, …). The agent gets the issue and a shell in a Docker container
with the repo checked out, explores and edits the code, and the hidden tests decide
if the bug is fixed. Contexts routinely reach 30k–100k tokens, so this exercises
long-context prefill, prompt caching and decode speed at depth, not just short prompts.

| `SWE_SUBSET` | Tasks | Use |
|---|---|---|
| `smoke` | 1 (`sympy__sympy-15599`) | simple single fix-a-bug task; checks the whole pipeline (`quick`) |
| `mini` | 50 ([SWE-bench Verified Mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)) | default for `standard` |
| `verified` | 500 | comparable with the [SWE-bench bash-only leaderboard](https://www.swebench.com/) (`full`, takes days) |

The agent uses mini-swe-agent's own `swebench.yaml` (the setup behind the bash-only
leaderboard). Only the model block changes: the local server, Qwen's recommended
sampling, and a step limit (`SWE_STEP_LIMIT`, default 100).

## Suites

```
./run.sh quick | standard | full
./run.sh perf | ref4167 | server | mlx | coding | agent [smoke|mini|verified]
RUN_DIR=results/<run> ./run.sh coding     # add a suite to an existing run
./run.sh collect results/<run>            # rebuild summary.md / summary.json
```

All settings are in [`config.env`](config.env). Put overrides in `config.local.env`
(git-ignored), for example:

```bash
GGUF_QUANTS="UD-Q4_K_XL"      # primary quant only
SWE_SUBSET=verified
AGENT_THINKING=0              # faster agent runs, usually lower resolve rate
MEASURE_POWER=1               # run `sudo -v` before ./run.sh
```

## Defaults and why

- **Engine:** llama.cpp built from source with Metal (`-ngl 999 -fa on --jinja`), KV cache
  `q8_0`, 128k server context. MLX runs as a comparison.
- **Quants:** `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL (~18 GB, primary) and Q8_0 (~29 GB).
  Both fit in 64 GB with room for a long KV cache.
- **MTP:** Qwen3.8 GGUFs ship a multi-token-prediction head. If the llama.cpp build
  supports `--spec-type draft-mtp`, the server, coding and agent suites use it
  (`MTP_DRAFT_N=2`) and the latency suite reports MTP on and off. `llama-bench`
  numbers are always plain (no speculation), which is what the community tables use.
- **Sampling:** the latency suite turns thinking off and forces fixed output lengths so
  the runs are comparable. The agent uses Qwen's recommended thinking-mode
  sampling (temp 1.0, top-p 0.95, top-k 20). EvalPlus is greedy with thinking off,
  as on its leaderboard.
- **Docker:** SWE-bench images are x86_64 only. They run under Rosetta through Colima
  (installed by `bootstrap.sh` when no Docker is running) or Docker Desktop/OrbStack
  with Rosetta enabled.

## Rough time and disk budget

These are estimates, not measurements. Replace them with your own after the first run.

| Item | Estimate |
|---|---|
| Disk | ~18 GB Q4 + ~29 GB Q8 + ~16 GB MLX + ~24 GB Llama-2 reference + ~1–3 GB per SWE-bench image |
| `quick` | 30–90 min (most of it is the one SWE-bench task and its image pull) |
| SWE-bench Verified Mini | roughly 12–30 h with thinking on at single-digit-to-low-teens tok/s decode |
| SWE-bench Verified (500) | several days |

## Tips for clean numbers

- Plug in the Mac, close other apps, and turn off Low Power Mode (System Settings → Energy).
  The run uses `caffeinate` so the Mac stays awake.
- For Q8_0 with very long contexts, raise the GPU memory limit for the session:
  `./bootstrap.sh --no-download --wired-limit` (sudo, reset at reboot).
- Run each suite twice and compare. `llama-bench` reports ± stddev over `BENCH_REPS`.

## Layout

```
bootstrap.sh            one-time setup (idempotent)
run.sh                  suites, results collection
config.env              all settings
bench/server_bench.py   TTFT/TPOT/ITL/prefill/cache/concurrency client (OpenAI API)
bench/telemetry_proxy.py  logs llama.cpp timings for every agent request
bench/swe_config.py     mini-swe-agent config for the local model
bench/collect.py        builds summary.md / summary.json
bench/sysinfo.py        env.json
results/<run>/          raw JSON/logs + summary.md (trajectories and eval logs are git-ignored)
```
