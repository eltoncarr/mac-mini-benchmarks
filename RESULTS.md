# Qwen3.8-27B on a Mac mini M5 Pro (64 GB): IQ3_S-mtp vs Q8_0

Results from the `final` protocol of this repo, run on 10–11 October 2026.
Raw data: [`results/20261010-1516_Qwen3.8-27B_gsq-iq3s-mtp_final`](results/20261010-1516_Qwen3.8-27B_gsq-iq3s-mtp_final),
[`results/20261010-1905_Qwen3.8-27B_q8_0_final`](results/20261010-1905_Qwen3.8-27B_q8_0_final)
and [`results/compare-20261011-1231.md`](results/compare-20261011-1231.md).

## Summary

On this machine the 12 GB **ISTA-DASLab GSQ-RCO IQ3_S-mtp** build scores the same on
HumanEval+ as the 29 GB **Q8_0** (89.6% pass@1 each) and generates **2.0× faster** without
speculative decoding, **1.3–1.4× faster** with MTP, and **1.6× faster at 64k context**.
Prompt processing (prefill) is the same for both: ~400 t/s, falling to 190 t/s (llama-bench)
to 250 t/s (through the server) at 64k context.

| | IQ3_S-mtp | Q8_0 | IQ3_S vs Q8_0 |
|---|---:|---:|---:|
| file size | 11.3 GiB | 27.1 GiB | 2.4× smaller |
| HumanEval / **HumanEval+** (pass@1, greedy) | 93.3% / **89.6%** | 92.1% / **89.6%** | same |
| generation, llama-bench tg128 | 20.0 t/s | 10.1 t/s | 1.98× |
| generation at 64k context, tg128 @ d65536 | 13.3 t/s | 8.3 t/s | 1.61× |
| short chat (256 tokens), MTP on | 28.9 t/s | 22.6 t/s | 1.28× |
| API generation (llama-benchy), MTP on | 24.6 t/s | 17.4 t/s | 1.41× |
| prefill, pp512 | 400 t/s | 415 t/s | 0.96× |
| 4 parallel requests, total generation | 41.0 t/s | 33.1 t/s | 1.24× |
| energy per generated token, short chat, MTP on | ~1.2 J | ~1.5 J | ~20% less |

## Setup

| | |
|---|---|
| machine | Mac mini (Mac17,16), Apple M5 Pro, 6 P + 12 E CPU cores, 20 GPU cores, 64 GB, 307 GB/s |
| OS | macOS 27.0, GPU wired-memory limit 48 GB (default) |
| engine | llama.cpp `d4d82d67f` (b11515), Metal, `-ngl 999 -fa on --jinja`, KV cache `q8_0`, server context 131072 |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 2` (both GGUFs carry the MTP head) |
| tools | llama-bench, llama-benchy 0.4.0, EvalPlus 0.3.1, powermetrics |
| models | [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`; [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-Q8_0.gguf` |
| reproduce | `./run.sh final` then `MODEL=q8_0 ./run.sh final` (see [README](README.md#final-run-publishable-results)) |

## Coding quality: HumanEval+ (EvalPlus)

Greedy decoding, thinking off, 164 problems, scored with EvalPlus's base and extended tests.

| model | HumanEval (base tests) | HumanEval+ (base + extra tests) |
|---|---:|---:|
| IQ3_S-mtp | 93.3% | 89.6% |
| Q8_0 | 92.1% | 89.6% |

The difference on base tests is two problems out of 164, which is within noise for a single
greedy run. This agrees with ISTA-DASLab's description of the IQ3_S build as task-lossless; it
is one benchmark, not a full quality evaluation.

## Generation and prefill vs context (llama-bench)

Engine only, no server, no speculative decoding; 5 repetitions, ± below 1%.

| context depth | IQ3_S pp512 | Q8_0 pp512 | IQ3_S tg128 | Q8_0 tg128 | tg ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 400.1 | 414.8 | 19.96 | 10.12 | 1.97× |
| 4k | 339.4 | 352.8 | 19.06 | 9.97 | 1.91× |
| 16k | 300.9 | 306.2 | 16.94 | 9.60 | 1.76× |
| 32k | 254.9 | 255.8 | 15.69 | 9.14 | 1.72× |
| 64k | 190.0 | 191.7 | 13.34 | 8.28 | 1.61× |

Generation is limited by memory bandwidth: Q8_0 reads its 27 GiB of weights for every token
and reaches ~290 GB/s of the chip's 307 GB/s. IQ3_S reads 2.4× less but spends compute
unpacking 3-bit weights, so it gains 2×, not 2.4×. At long context, attention over the KV
cache costs both models the same, which shrinks the gap.

## Through the server API, MTP off vs on (llama-benchy)

Generation speed (t/s) after a 512-token prompt, every request uncached.

| context depth | IQ3_S | IQ3_S + MTP | Q8_0 | Q8_0 + MTP |
|---:|---:|---:|---:|---:|
| 0 | 19.3 | 24.6 | 10.1 | 17.4 |
| 4k | 18.6 | 20.2 | 9.9 | 17.3 |
| 16k | 16.4 | 19.9 | 9.5 | 16.0 |
| 32k | 14.8 | 15.4 | 9.0 | 14.0 |
| 64k | 12.7 | 13.6 | 8.1 | 10.7 |

MTP helps Q8_0 far more (+30–75%) than IQ3_S (+4–27%): Q8_0 leaves the GPU waiting on memory,
so verifying draft tokens is nearly free, while IQ3_S has less spare compute. These prompts are
book text; on code, MTP's speedup is larger (short chat: IQ3_S 18.3 → 28.9, Q8_0 10.1 → 22.6).

## Time to first token: uncached vs prefix cache

Prompt of 512 new tokens on top of a context of the given size.

| context | IQ3_S, uncached | IQ3_S, context cached | Q8_0, uncached | Q8_0, context cached |
|---:|---:|---:|---:|---:|
| 4k | 13.0 s | 1.7 s | 12.3 s | 1.6 s |
| 16k | 50.8 s | 1.9 s | 49.2 s | 1.9 s |
| 32k | 110.9 s | 2.4 s | 107.9 s | 2.3 s |
| 64k | 264.6 s | — | 259.4 s | — |

A fresh 64k-token prompt takes about 4.5 minutes before the first token on either model.
Chat clients and coding agents resend the same growing conversation, so llama-server's prompt
cache turns that into ~2 seconds per turn.

## Parallel requests (llama-benchy, 4 server slots, MTP off)

Total generation across all requests, short prompts (512 tokens, no prior context).

| parallel requests | IQ3_S | Q8_0 |
|---:|---:|---:|
| 1 | 18.7 t/s | 10.1 t/s |
| 2 | 24.0 t/s | 18.2 t/s |
| 4 | 41.0 t/s | 33.1 t/s |

Batching shares one read of the weights across requests, so total throughput grows almost
linearly for Q8_0. With 4k of context per request the gain disappears (prefill of the other
requests interrupts decoding), so this applies to short requests such as several light agents.

## Power (powermetrics, CPU + GPU + ANE)

| phase | IQ3_S avg W | Q8_0 avg W |
|---|---:|---:|
| llama-bench | 35.6 | 29.4 |
| server, MTP off | 35.4 | 33.6 |
| server, MTP on | 36.0 | 34.7 |

Energy per generated token, from server-phase average power and short-chat decode speed
(rough: the average also covers prefill):

| | IQ3_S | Q8_0 |
|---|---:|---:|
| MTP off | ~1.9 J | ~3.3 J |
| MTP on | ~1.2 J | ~1.5 J |

## Observed during exploratory agent runs (not part of the final protocol)

SWE-bench Verified tasks with mini-swe-agent (thinking on, MTP on) were run during harness
development on the same build. They are too few to compare the two models' quality (IQ3_S
resolved 4 of 6 tasks, Q8_0 4 of 6), but they show realistic speed in agent work, where the
context grows to 50k–130k tokens and the prompt cache hit ratio is ~99%:

| context | IQ3_S decode | Q8_0 decode |
|---|---:|---:|
| 0–8k | 26 t/s | 21 t/s |
| 32–64k | 17 t/s | 15 t/s |
| 64k+ | 15 t/s | 12 t/s |

MTP draft acceptance in those runs was 73–76%. A task took about 1 hour (IQ3_S) to 2 hours (Q8_0).

## Caveats

- One machine, one run per model; llama-bench repetitions vary by under 1%, other suites were
  not repeated.
- HumanEval+ is a single greedy run with thinking off; it says nothing about long-horizon or
  agentic quality.
- Power is the average over whole phases, so energy per token is approximate.
- Results are specific to llama.cpp `d4d82d67f`; MTP and Metal kernels change between versions.

## llama.cpp Apple Silicon reference (Discussion #4167)

Pending: the Llama-2 7B F16 reference model was missing when `./run.sh ref4167` ran. The
summary row for the thread will be added here once it has been run.
