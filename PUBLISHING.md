# Where to publish results

Each venue has its own method. Post the numbers in the format it expects and link
back to `results/<run>/summary.md` (or this repo, if you make it public) for the raw data.

## Inference speed

| Venue | What to post | How |
|---|---|---|
| [llama.cpp Discussion #4167 "Performance of llama.cpp on Apple Silicon M-series"](https://github.com/ggml-org/llama.cpp/discussions/4167) | `results/<run>/llama_bench/4167.md` (Llama-2 7B F16/Q8_0/Q4_0, pp512/tg128) | Reply in the thread with chip, GPU cores, RAM, and the table plus `build:` line. This is the main table for comparing Macs. The thread pins a llama.cpp commit for the reference numbers; set `LLAMA_CPP_REF` to it for an exact comparison. M5 Pro rows already exist for MacBook Pro, so a Mac mini row is new data. |
| [unsloth/Qwen3.8-27B-GGUF discussions](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions) | `llama-bench` table for Qwen3.8-27B per quant, MTP on/off decode t/s | Community threads where users share speeds by hardware. |
| [LocalScore](https://www.localscore.ai/) | LocalScore's own test (built on llama-bench) | Accepts results only from its official CLI (`localscore` from Mozilla's llamafile releases). Run it on its own; it isn't wrapped here. |
| [oMLX community benchmarks](https://omlx.ai/benchmarks/performance) and [mlx-Chronos](https://github.com/jundot/omlx/discussions/1391) | MLX prefill/decode, TTFT | For MLX-engine numbers. mlx-Chronos keeps a leaderboard of MLX serving metrics (TTFT, throughput) and asks for results from more Apple silicon machines. |
| r/LocalLLaMA | `summary.md` | Informal, but where most Mac-mini-for-LLM comparisons get read. Include quant, context, llama.cpp commit and whether MTP was on. |

## Model quality

| Venue | What to post | How |
|---|---|---|
| [SWE-bench leaderboards](https://www.swebench.com/) (bash-only / mini-swe-agent) | Full SWE-bench Verified (500) run: `preds.json` + trajectories | Submissions go through a PR to [SWE-bench/experiments](https://github.com/SWE-bench/experiments) and must be the full split with trajectories; Verified Mini (50) and smoke runs are not accepted. The bash-only board is run by the SWE-bench team on the same mini-swe-agent setup, so a local quantized Qwen3.8 result is comparable but should be labeled with its quant. |
| [EvalPlus leaderboard](https://evalplus.github.io/leaderboard.html) | HumanEval+/MBPP+ pass@1 | Open an issue on [evalplus/evalplus](https://github.com/evalplus/evalplus) with the samples file from `results/<run>/evalplus/raw/`. They usually list full-precision models; a quantized local result is better posted alongside the speed numbers. |

## Checklist for any post

- Hardware: Mac mini, M5 Pro, CPU/GPU core count, 64 GB, macOS version (`env.json`)
- Software: llama.cpp commit, engine flags (`-fa on`, KV `q8_0`, context), MTP on/off
- Model: exact repo + quant file (e.g. `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL)
- For SWE-bench: subset, step limit, thinking on/off, evaluator (local harness vs sb-cli)
