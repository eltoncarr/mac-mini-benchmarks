# vLLM on the Mac mini M5 Pro (vllm-metal)

The same benchmark protocol as the llama.cpp harness, run on
[vLLM](https://github.com/vllm-project/vllm) with the
[vllm-metal](https://github.com/vllm-project/vllm-metal) plugin (vLLM's scheduler, paged
KV cache and OpenAI server; MLX and Metal kernels underneath). The question it answers: on
the same 64 GB Mac mini with the same Qwen3.8-27B, what does vLLM's serving stack give
over llama.cpp, especially with several concurrent requests, and does quality hold?

Results land in the same `results/<run>/` layout and go through the same `collect.py`, so
llama.cpp and vLLM runs can be compared directly:

```bash
vllm/bootstrap.sh                    # vLLM + vllm-metal (pinned wheels), MLX models
vllm/run.sh quick                    # pass 1: mlx-4bit (first in VLLM_MODELS)
MODEL=mlx-8bit vllm/run.sh quick     # pass 2, optional
./run.sh compare results/*_gsq-iq3s-mtp_final results/*_vllm-*_quick
```

## What differs from the llama.cpp harness, and why

These are limits of vllm-metal 0.31 for Qwen3.8 (a hybrid model: full attention plus
Gated DeltaNet linear-attention layers), taken from its docs:

| | llama.cpp harness | vLLM harness |
|---|---|---|
| weights | GGUF: GSQ IQ3_S-mtp (12 GB), Q8_0 (29 GB) | MLX: `mlx-community/Qwen3.8-27B-4bit` (~15 GB), `-8bit` (~29 GB). vllm-metal's GGUF loader takes only dense Qwen2/Qwen3/Llama/Mistral and no IQ quants. |
| engine-only speed | `llama-bench` pp512/tg128 at depth | not available; `vllm bench serve` with one request at a time (prompt = depth + 512, 128 out, greedy, `ignore_eos`), so the speeds include the HTTP server |
| speculative decoding | MTP (`draft-mtp`, 2 tokens) | **n-gram** (3 tokens): MTP is Gemma 4 only on Metal, and draft models are rejected on hybrid targets. Drafts only greedy requests; needs `--no-async-scheduling` and FP32 GDN state. Quality suites run without it. |
| KV cache | `q8_0` | model dtype (`VLLM_KV_CACHE_DTYPE=auto`); only the full-attention layers keep KV |
| prefix cache | on by default | experimental for Qwen3.8 in vllm-metal; off except in the prefix-cache step and the agent suite |
| concurrency | llama-benchy c1/2/4 | the same llama-benchy runs, plus a `vllm bench serve` sweep at 1/2/4/8 (`VLLM_CONCURRENCY_LEVELS`) |

What stays the same: llama-benchy (pp/tg at depth over the API, TTFT, prefix cache,
concurrency), `bench/server_bench.py` short-chat latency (run greedy here so the n-gram arm
drafts), EvalPlus HumanEval+ (greedy, thinking off), mini-swe-agent on SWE-bench with the
telemetry proxy, powermetrics, and `env.json` (plus vLLM, vllm-metal, MLX and mlx-lm versions).

## Suites

```
[MODEL=<label>] vllm/run.sh quick | final | standard | full
[MODEL=<label>] vllm/run.sh perf | server | coding | agent [smoke|mini|verified]
RUN_DIR=results/<run> vllm/run.sh coding     # add a suite to an existing run
```

| suite | contents |
|---|---|
| `perf` | `vllm bench serve` sweep over `BENCH_DEPTHS`, one request at a time → `vllm_bench/*-perf_*.json` |
| `server` | llama-benchy + short-chat latency, plain and n-gram; full: prefix cache, concurrency (llama-benchy c1/2/4 and `vllm bench serve` c1/2/4/8) |
| `coding` | EvalPlus HumanEval+ (MBPP+ in `full`) |
| `agent` | mini-swe-agent on SWE-bench (needs Docker with Rosetta, set up by the repo's `./bootstrap.sh`) |
| `quick` | perf at depth 0/16k, server at 0/16k, one SWE-bench task |
| `final` | perf + full server suite + HumanEval+: the publishable set, as `./run.sh final` |

After each server phase the run saves vLLM's Prometheus `/metrics`
(`server/*.metrics.txt`); `summary.md` reports n-gram acceptance and prefix-cache hit ratio from them.

## Settings

All in [`config.env`](config.env), on top of the repo's [`config.env`](../config.env)
(shared: `MODEL_NAME`, `BENCH_DEPTHS`, `BENCHY_*`, `LATENCY_REPS`, `EVALPLUS_*`, `SWE_*`,
`MEASURE_POWER`). Overrides for both go in the repo's `config.local.env`. Main knobs:
`VLLM_METAL_VERSION` / `VLLM_VERSION` (pinned wheels; `vllm/bootstrap.sh --reinstall` after
a change), `VLLM_MODELS`, `VLLM_MAX_MODEL_LEN` (128k), `VLLM_GPU_MEM_UTIL` (0.75),
`VLLM_MAX_BATCHED_TOKENS` (2048), `USE_NGRAM`, `NGRAM_SPEC_CONFIG`.

vLLM and the plugin live in their own venv (`vendor/vllm-metal-venv`) so their pinned
MLX/torch versions never touch the client tools in `.venv`. The 4-bit MLX model is the same
download `./run.sh mlx` uses (`models/mlx/`).

## Other models: Meta Muse Glimmer

[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (dense, Apache 2.0,
DFlash drafter) fits this machine (~17–20 GB at 4-bit), but it is not in vllm-metal's
supported-model list yet. vllm-metal takes its model layers from mlx_lm, and its DFlash path is
qualified only for one Qwen3-4B pair. The `config.env` comment shows the three lines to try it
(`RadixArk/Muse-Glimmer-q4k-dynamic-MLX`); expect it to fail at load until mlx_lm supports
the architecture. On the llama.cpp side it needs a llama.cpp master build that includes
Glimmer support. Muse Spark Open (816B MoE) needs hundreds of GB and does not fit.

## Before a final run

Same as for llama.cpp: plug in, close other apps (browsers keep the GPU busy), and leave the
Mac alone. vllm-metal's [benchmarking guide](https://github.com/vllm-project/vllm-metal/blob/main/docs/benchmarking-macos.md)
also asks for alternating A/B order when comparing two configurations, and for recording
the vLLM / vllm-metal / MLX versions (in `env.json`) and server logs (in `logs/`).
