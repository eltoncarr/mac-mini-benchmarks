# Mac mini M5 Pro LLM benchmarks: Qwen3.8-27B on llama.cpp

Reproducible local-LLM benchmarks for an **Apple Mac mini M5 Pro (20-core GPU, 64 GB)** with
**llama.cpp (Metal)**, focused on **Qwen3.8-27B** (dense, Apache 2.0).

**Results (October 2026):**

- **Generation:** 20.0 tokens/s (IQ3_S-mtp) and 10.1 tokens/s (Q8_0); **28.9 and 22.6 tokens/s with MTP**
  speculative decoding on code; 13.3 and 8.3 tokens/s at 64k context.
- **Quality:** both builds score **89.6% on HumanEval+**, so the 12 GB IQ3_S-mtp is as good as the 29 GB Q8_0
  on this test while being 1.3–2× faster.
- **Prefill:** ~400 tokens/s, falling to 190–250 tokens/s at 64k context; with the prompt cache, follow-up
  turns at 32k context start in ~2 s.
- **Hardware reference:** on llama.cpp's standard Llama-2 7B test the Mac mini matches the M5 Pro MacBook Pro.

Full write-up: **[RESULTS.md](RESULTS.md)**. Machine-readable data (one row per measurement):
[`data/qwen3.8-27b-mac-mini-m5-pro.json`](data/qwen3.8-27b-mac-mini-m5-pro.json) and
[`.csv`](data/qwen3.8-27b-mac-mini-m5-pro.csv).

Models compared:

| label | file | size | what it is |
|---|---|---:|---|
| `gsq-iq3s-mtp` (default) | [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | ~12 GB | GSQ+RCO non-uniform 3.5 bpw, with the MTP head for speculative decoding |
| `q8_0` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `*-Q8_0.gguf` | ~29 GB | standard 8-bit, near-lossless baseline |

The question it answers: how much speed does the 12 GB build gain on this machine
(smaller weights, plus MTP), and how much coding and agentic quality does it lose
compared with Q8_0?

Each `./run.sh` pass benchmarks **one model**, in its own results folder. Run the
baseline as a second pass whenever you want it, then put the passes side by side:

```bash
./run.sh quick                          # pass 1: gsq-iq3s-mtp (first in GGUF_MODELS)
MODEL=q8_0 ./run.sh quick               # pass 2, optional: the Q8_0 baseline
./run.sh compare results/*_quick        # -> results/compare-<date>.md
```

One model per pass keeps runs isolated (a crash or reboot costs one run, not both),
lets you stop after the first model, and gives each folder one model, one
llama.cpp commit and one `env.json`. The compare step warns if the passes used
different llama.cpp commits or different suites. Models are listed in
`GGUF_MODELS` in [`config.env`](config.env), so adding another build is one line.

The suite covers raw inference speed, serving latency, standard coding scores and agentic bug fixing on real repositories, all with
standard tools so the numbers can be posted where the community compares them
(see [PUBLISHING.md](PUBLISHING.md)).

```bash
git clone https://github.com/eltoncarr/mac-mini-benchmarks && cd mac-mini-benchmarks
./bootstrap.sh            # one-time setup: tools, llama.cpp (Metal), Python env, models, Docker
./run.sh quick            # 30-90 min: speed + API latency + one SWE-bench bug fix
./run.sh standard         # overnight or longer: everything below
cat results/*/summary.md
```

## Final run (publishable results)

`./run.sh final` is the fixed protocol for published numbers: the full llama-bench depth
sweep (0–64k), the full server suite (llama-benchy plain and MTP, prefix cache,
concurrency 1/2/4, short-chat latency) and HumanEval+ (EvalPlus, greedy, thinking off).
SWE-bench and MLX are not part of it.

```bash
git pull && git log -1 --oneline           # note the harness commit for the write-up
# config.local.env: keep only what the final run needs
cat > config.local.env <<'CFG'
RUN_4167_REFERENCE=1
MEASURE_POWER=1
CFG
./bootstrap.sh --no-docker --no-mlx        # adds the Llama-2 7B reference models; keeps the llama.cpp build
./run.sh ref4167                           # ~15 min, once per machine
./run.sh final                             # ~4-5 h  (gsq-iq3s-mtp)
MODEL=q8_0 ./run.sh final                  # ~5-6 h
./run.sh compare $(printf '%s\n' results/*_final | tail -n2)
```

Before starting: plug in, close other apps (browsers too: they keep the GPU busy and
appear in `env.json` as sleep blockers), and leave the Mac alone until it finishes. With
`MEASURE_POWER=1` the run asks for your password once and keeps sudo active.
Check `problems.log` in each run folder: an empty file means every step ran.

## What gets measured

| Suite | Tool (standard) | Metrics |
|---|---|---|
| `perf` | `llama-bench` | **pp512**, **tg128** (t/s ± stddev), plus pp/tg at context depth 4k/16k/32k/64k (`-d`) for each model |
| `ref4167` (run once) | `llama-bench` on Llama-2 7B F16/Q8_0/Q4_0 | the exact command the [llama.cpp Apple Silicon thread](https://github.com/ggml-org/llama.cpp/discussions/4167) uses, so the chip can be compared with every other Mac |
| `server` | [llama-benchy](https://github.com/eugr/llama-benchy) against `llama-server` | the same pp/tg at depth through the OpenAI API, plus TTFT (time to first response, estimated prompt processing, end-to-end TTFT), prefix-cache reuse and 1/2/4 concurrent requests; with and without **MTP** speculative decoding, per model |
| | `bench/server_bench.py` | what llama-benchy doesn't report: inter-token latency p50/p90/p99, TPOT and e2e latency for a short chat request |
| `mlx` (run once) | `mlx_lm.benchmark`, then llama-benchy against `mlx_lm.server` | prompt/generation t/s and peak memory for the MLX 4-bit build, and the same API numbers as llama.cpp |
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
| `smoke` | 1 (`sympy__sympy-15599`) | simple single fix-a-bug task per model; checks the whole pipeline (`quick`) |
| `mini` | 50 ([SWE-bench Verified Mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)) | default for `standard` |
| `verified` | 500 | comparable with the [SWE-bench bash-only leaderboard](https://www.swebench.com/) (`full`, takes days) |

`SWE_SLICE="0:10"` runs only part of `mini` or `verified` (here the first 10 tasks). With
thinking on, the smoke task took about 1 h on IQ3_S-mtp (42 model calls, up to 78k context),
so a slice of 5–10 tasks per model is a practical overnight run.

The agent uses mini-swe-agent's own `swebench.yaml` (the setup behind the bash-only
leaderboard). Only the model block changes: the local server, Qwen's recommended
sampling, and a step limit (`SWE_STEP_LIMIT`, default 100).

## Suites

```
[MODEL=<label>] ./run.sh quick | standard | full
[MODEL=<label>] ./run.sh perf | server | coding | agent [smoke|mini|verified]
./run.sh ref4167 | mlx                    # machine-level, not tied to a GGUF model; run once
RUN_DIR=results/<run> ./run.sh coding     # add a suite to an existing run
./run.sh collect results/<run>            # rebuild summary.md / summary.json
./run.sh compare results/<run> ...        # side-by-side table of several runs
```

`standard` and `full` no longer include `ref4167` and `mlx`, since those don't
depend on the model being tested.

All settings are in [`config.env`](config.env). Put overrides in `config.local.env`
(git-ignored), for example:

```bash
MODEL=q8_0                    # default model for ./run.sh (else the first in GGUF_MODELS)
SWE_SUBSET=verified
AGENT_THINKING=0              # faster agent runs, usually lower resolve rate
MEASURE_POWER=1               # run `sudo -v` before ./run.sh
```

## Defaults and why

- **Engine:** llama.cpp built from source with Metal (`-ngl 999 -fa on --jinja`), KV cache
  `q8_0`, 128k server context. MLX runs as a comparison.
- **Models:** see the table at the top. Q8_0 plus a 128k KV cache fits in 64 GB; the
  12 GB IQ3_S leaves plenty of room.
- **MTP:** if the llama.cpp build supports `--spec-type draft-mtp`, each model is tried
  with MTP (`MTP_DRAFT_N=2`). The run checks that the server really drafts (llama.cpp
  reports `draft_n`); a GGUF without an MTP head is simply measured without it. The
  server suite reports MTP on and off; coding and agent suites use MTP when it works,
  which is faster without changing the output distribution, and record whether it was on. `llama-bench`
  numbers are always plain (no speculation), which is what the community tables use.
  llama-benchy uses natural text (a Project Gutenberg book) for prompts, so MTP
  acceptance is realistic rather than inflated by repetitive filler.
- **llama-bench vs llama-benchy:** `llama-bench` measures the engine directly and is
  what the Apple Silicon tables use. llama-benchy measures the same pp/tg through the
  server API, which is what an app or agent actually sees, and works the same way for
  llama.cpp and MLX. It needs the model's tokenizer (`models/tokenizer`, set up by `bootstrap.sh`;
  built from `vocab.json` + `merges.txt` if the repo has no `tokenizer.json`, and checked
  against llama-server's own token count before each run) and downloads its book text once on first use.
- **Sampling:** the server suite turns thinking off and forces fixed output lengths so
  the runs are comparable. The agent uses Qwen's recommended thinking-mode
  sampling (temp 1.0, top-p 0.95, top-k 20). EvalPlus is greedy with thinking off,
  as on its leaderboard.
- **Docker:** SWE-bench images are x86_64 only. They run under Rosetta through Colima
  (installed by `bootstrap.sh` when no Docker is running) or Docker Desktop/OrbStack
  with Rosetta enabled. Check with `docker run --rm --platform linux/amd64 busybox:1.36 uname -m`
  (prints `x86_64`). If pulls fail with `docker-credential-desktop: executable file not found`,
  `~/.docker/config.json` still points at Docker Desktop's credential helper; `bootstrap.sh`
  removes that `credsStore` entry (backup in `config.json.bak`).

## Rough time and disk budget

These are estimates, not measurements. Replace them with your own after the first run.

| Item | Estimate |
|---|---|
| Disk | ~29 GB Q8_0 + ~12 GB IQ3_S + ~16 GB MLX + ~24 GB Llama-2 reference + ~1–3 GB per SWE-bench image |
| `quick` | 30–90 min per model (most of it is the SWE-bench task and its image pull) |
| SWE-bench Verified Mini | roughly 12–30 h per model with thinking on; Q8_0 is the slower of the two |
| SWE-bench Verified (500) | several days |

## Tips for clean numbers

- Plug in the Mac, close other apps, and turn off Low Power Mode (System Settings → Energy).
  The run uses `caffeinate` so the Mac stays awake.
- For Q8_0 with very long contexts, raise the GPU memory limit for the session:
  `./bootstrap.sh --no-download --wired-limit` (sudo, reset at reboot).
- For repeatability, `llama-bench` and llama-benchy already repeat each test (`BENCH_REPS`,
  `LATENCY_REPS`) and report ± stddev; a second identical pass is only needed to check
  run-to-run drift (thermals, background load).

## Layout

```
bootstrap.sh            one-time setup (idempotent)
run.sh                  suites, results collection
config.env              all settings
bench/server_bench.py   short-chat ITL percentiles / TPOT (complements llama-benchy)
bench/telemetry_proxy.py  logs llama.cpp timings for every agent request
bench/swe_config.py     mini-swe-agent config for the local model
bench/tokenizer_fix.py  builds/validates the tokenizer.json llama-benchy needs
bench/collect.py        builds summary.md / summary.json
bench/sysinfo.py        env.json
results/<run>/          raw JSON/logs + summary.md (trajectories and eval logs are git-ignored)
```

## Exporting data

`bench/export.py` turns final runs into the flat JSON/CSV in `data/` (one row per measurement:
model, metric, context, prompt size, concurrency, MTP, value, unit, tool):

```bash
python bench/export.py --out data/qwen3.8-27b-mac-mini-m5-pro \
  results/<iq3s final run> results/<q8_0 final run> --ref4167 results/<ref4167 run>
```

## License

Code is licensed under the [MIT License](LICENSE). Benchmark data and results (`data/`, `results/`,
`RESULTS.md`) are licensed under [CC BY 4.0](LICENSE-DATA); please credit
"Mac mini M5 Pro LLM benchmarks by eltoncarr" with a link to this repository.
