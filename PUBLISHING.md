# Where to publish results

Each venue has its own method. Post the numbers in the format it expects and link
back to `results/<run>/summary.md` (or this repo, if you make it public) for the raw data.

## Inference speed

| Venue | What to post | How |
|---|---|---|
| [llama.cpp Discussion #4167 "Performance of llama.cpp on Apple Silicon M-series"](https://github.com/ggml-org/llama.cpp/discussions/4167) | `results/<run>/llama_bench/4167.md` (Llama-2 7B F16/Q8_0/Q4_0, pp512/tg128) and the summary row in that run's `summary.md` | Reply in the thread with chip, GPU cores, RAM, the table plus `build:` line, and the one-line summary row. This is the main table for comparing Macs. The thread pins a llama.cpp commit for its reference numbers (`REF4167_LLAMA_REF`, default `c1d0e7a00`); `bootstrap.sh` builds llama-bench at that commit separately, so the main benchmark build is unchanged. The summary row records which commit was used. M5 Pro rows already exist for MacBook Pro, so a Mac mini row is new data; the ready-to-post text is in [RESULTS.md](RESULTS.md#post-for-the-thread). |
| [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF discussions](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/discussions) | The comparison table: IQ3_S-mtp vs Q8_0 speed (MTP on/off) and quality (HumanEval+, SWE-bench) | The GSQ authors and users discuss quality vs size there; Apple silicon results with a Q8_0 baseline on the same machine are directly useful. |
| [unsloth/Qwen3.8-27B-GGUF discussions](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions) | `llama-bench` table for Q8_0, and the llama-benchy table | Community threads where users share speeds by hardware. |
| Wherever llama-benchy tables are shared (e.g. the [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/new-tool-llama-benchy-llama-bench-style-benchmarking-for-any-llm-backend-vllm-sglang-llama-cpp-etc/356698) for DGX Spark, r/LocalLLaMA) | `results/<run>/benchy/*.json` rendered in `summary.md` | llama-benchy output is the common format for comparing server-side pp/tg at depth across machines and engines (vLLM, SGLang, llama.cpp, MLX). Include the llama-benchy version and latency mode. |
| [LocalScore](https://www.localscore.ai/) | LocalScore's own test (built on llama-bench) | Accepts results only from its official CLI (`localscore` from Mozilla's llamafile releases). Run it on its own; it isn't wrapped here. |
| [oMLX community benchmarks](https://omlx.ai/benchmarks/performance) and [mlx-Chronos](https://github.com/jundot/omlx/discussions/1391) | MLX prefill/decode, TTFT | For MLX-engine numbers. mlx-Chronos keeps a leaderboard of MLX serving metrics (TTFT, throughput) and asks for results from more Apple silicon machines. |
| r/LocalLLaMA | `summary.md` | Informal, but where most Mac-mini-for-LLM comparisons get read. Include quant, context, llama.cpp commit and whether MTP was on. |

## Model quality

| Venue | What to post | How |
|---|---|---|
| [SWE-bench leaderboards](https://www.swebench.com/) (bash-only / mini-swe-agent) | Full SWE-bench Verified (500) run: `preds.json` + trajectories | Submissions go through a PR to [SWE-bench/experiments](https://github.com/SWE-bench/experiments) and must be the full split with trajectories; Verified Mini (50) and smoke runs are not accepted. The bash-only board is run by the SWE-bench team on the same mini-swe-agent setup, so a local quantized Qwen3.8 result is comparable but should be labeled with its quant. |
| [Terminal-Bench 2.0 leaderboard](https://www.tbench.ai/leaderboard) | Full 89-task run: the Harbor job folder (`results/<run>/tbench/<model>/all/job/`) | Submissions go through the Terminal-Bench / Harbor submission process and must follow its rules (full task set, default timeouts, several attempts per task: set `TB_ATTEMPTS`). A local result on native arm64 images should say so; the per-machine Mac mini vs DGX Spark comparison is better posted with the speed numbers (r/LocalLLaMA, the NVIDIA DGX Spark forum). |
| [EvalPlus leaderboard](https://evalplus.github.io/leaderboard.html) | HumanEval+/MBPP+ pass@1 | Open an issue on [evalplus/evalplus](https://github.com/evalplus/evalplus) with the samples file from `results/<run>/evalplus/raw/`. They usually list full-precision models; a quantized local result is better posted alongside the speed numbers. |

## Checklist for any post

- Hardware: Mac mini, M5 Pro, CPU/GPU core count, 64 GB, macOS version, or DGX Spark, driver and CUDA version (`env.json`)
- Software: llama.cpp commit, engine flags (`-fa on`, KV `q8_0`, context), MTP on/off
- Models: exact repo + file for each (listed at the top of `summary.md`)
- For SWE-bench: subset, step limit, thinking on/off, evaluator (local harness vs sb-cli)
- For Terminal-Bench: Harbor version, agent, attempts, timeout multiplier, native vs published images, tasks at once
