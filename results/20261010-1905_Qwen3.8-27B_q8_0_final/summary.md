# Qwen3.8-27B on Mac mini (Apple M5 Pro, 64 GB)

Run `20261010-1905_Qwen3.8-27B_q8_0_final` · macOS 27.0 · GPU cores 20 · llama.cpp `d4d82d67f`

## Comparison

| model | size GiB | pp512 | tg128 | tg128 @ d65536 | chat tg | chat tg MTP | API tg | API tg MTP | API TTFT ms | HumanEval+ | SWE-bench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q8_0 | 27.05 | 414.7 | 10.1 | 8.3 | 10.1 | 22.6 | 10.1 | 17.4 | 1407 | 89.6% | — |

pp/tg in tokens/s. llama-bench columns (pp512, tg128) are the engine alone. chat tg is the short-chat stream (256 tokens, thinking off) through llama-server; API columns are llama-benchy. MTP columns are empty when MTP did not run.

- `q8_0`: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-Q8_0.gguf`

## llama-bench (llama.cpp, Metal)

| model | size GiB | test | t/s | ± |
|---|---:|---|---:|---:|
| q8_0 | 27.04 | pp512 | 414.75 | 0.29 |
| q8_0 | 27.04 | tg128 | 10.12 | 0.01 |
| q8_0 | 27.04 | pp512 @ d4096 | 352.81 | 1.40 |
| q8_0 | 27.04 | tg128 @ d4096 | 9.97 | 0.02 |
| q8_0 | 27.04 | pp512 @ d16384 | 306.23 | 1.33 |
| q8_0 | 27.04 | tg128 @ d16384 | 9.60 | 0.02 |
| q8_0 | 27.04 | pp512 @ d32768 | 255.81 | 1.02 |
| q8_0 | 27.04 | tg128 @ d32768 | 9.14 | 0.02 |
| q8_0 | 27.04 | pp512 @ d65536 | 191.74 | 2.18 |
| q8_0 | 27.04 | tg128 @ d65536 | 8.28 | 0.00 |

build `d4d82d67f`

## llama-benchy (OpenAI API, llama-bench style)

pp/tg in t/s (mean ± std); latencies in ms. TTFR = time to first response, est_ppt = estimated prompt processing time, e2e_ttft = end-to-end time to first token.

| config | test | t/s | ± | peak t/s | TTFR | est_ppt | e2e_ttft |
|---|---|---:|---:|---:|---:|---:|---:|
| llama.cpp-q8_0-concurrency | pp512 | 415.43 | 1.92 | — | 1396 | 1234 | 1396 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) | 10.09 | 0.00 | 11.00 | | | |
| llama.cpp-q8_0-concurrency | pp512 (c2) | 367.72 | 5.82 | — | 2792 | 2629 | 2792 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) (c2) | 18.21 | 0.02 | 20.00 | | | |
| llama.cpp-q8_0-concurrency | pp512 (c4) | 361.90 | 6.67 | — | 5671 | 5509 | 5671 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) (c4) | 33.07 | 0.01 | 36.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 | 395.83 | 2.33 | — | 5339 | 5177 | 5339 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) | 10.03 | 0.01 | 11.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 (c2) | 368.35 | 0.16 | — | 11012 | 10850 | 11012 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) (c2) | 17.76 | 0.01 | 19.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 (c4) | 353.67 | 0.09 | — | 21105 | 20943 | 21105 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) (c4) | 23.39 | 0.01 | 32.00 | | | |
| llama.cpp-q8_0-concurrency | pp512 @ d4096 | 382.14 | 0.84 | — | 12223 | 12061 | 12223 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) @ d4096 | 9.91 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0-concurrency | pp512 @ d4096 (c2) | 359.39 | 0.19 | — | 24657 | 24495 | 24657 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) @ d4096 (c2) | 15.59 | 0.00 | 17.00 | | | |
| llama.cpp-q8_0-concurrency | pp512 @ d4096 (c4) | 351.97 | 0.04 | — | 44255 | 44093 | 44255 |
| llama.cpp-q8_0-concurrency | tg128 (after pp512) @ d4096 (c4) | 12.63 | 0.00 | 26.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 @ d4096 | 376.00 | 0.52 | — | 16507 | 16345 | 16507 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) @ d4096 | 9.86 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 @ d4096 (c2) | 358.46 | 0.58 | — | 31135 | 30972 | 31135 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) @ d4096 (c2) | 12.24 | 0.00 | 16.00 | | | |
| llama.cpp-q8_0-concurrency | pp2048 @ d4096 (c4) | 353.49 | 0.21 | — | 51696 | 51533 | 51696 |
| llama.cpp-q8_0-concurrency | tg128 (after pp2048) @ d4096 (c4) | 8.84 | 0.00 | 21.00 | | | |
| llama.cpp-q8_0-mtp | pp512 | 390.02 | 5.86 | — | 1480 | 1316 | 1480 |
| llama.cpp-q8_0-mtp | tg128 (after pp512) | 17.43 | 1.04 | 17.67 | | | |
| llama.cpp-q8_0-mtp | pp2048 | 370.93 | 2.64 | — | 5689 | 5524 | 5689 |
| llama.cpp-q8_0-mtp | tg128 (after pp2048) | 18.83 | 1.47 | 19.33 | | | |
| llama.cpp-q8_0-mtp | pp512 @ d4096 | 360.17 | 0.73 | — | 12962 | 12798 | 12962 |
| llama.cpp-q8_0-mtp | tg128 (after pp512) @ d4096 | 17.31 | 2.22 | 18.00 | | | |
| llama.cpp-q8_0-mtp | pp2048 @ d4096 | 355.63 | 0.25 | — | 17443 | 17279 | 17443 |
| llama.cpp-q8_0-mtp | tg128 (after pp2048) @ d4096 | 16.53 | 1.69 | 17.33 | | | |
| llama.cpp-q8_0-mtp | pp512 @ d16384 | 329.06 | 0.37 | — | 51512 | 51348 | 51512 |
| llama.cpp-q8_0-mtp | tg128 (after pp512) @ d16384 | 16.02 | 1.77 | 16.67 | | | |
| llama.cpp-q8_0-mtp | pp2048 @ d16384 | 325.51 | 0.08 | — | 56793 | 56628 | 56793 |
| llama.cpp-q8_0-mtp | tg128 (after pp2048) @ d16384 | 14.18 | 0.73 | 14.67 | | | |
| llama.cpp-q8_0-mtp | pp512 @ d32768 | 294.81 | 0.04 | — | 113052 | 112887 | 113052 |
| llama.cpp-q8_0-mtp | tg128 (after pp512) @ d32768 | 14.01 | 2.07 | 14.33 | | | |
| llama.cpp-q8_0-mtp | pp2048 @ d32768 | 292.78 | 0.20 | — | 119082 | 118917 | 119082 |
| llama.cpp-q8_0-mtp | tg128 (after pp2048) @ d32768 | 14.35 | 0.30 | 15.00 | | | |
| llama.cpp-q8_0-mtp | pp512 @ d65536 | 244.24 | 0.34 | — | 270592 | 270428 | 270592 |
| llama.cpp-q8_0-mtp | tg128 (after pp512) @ d65536 | 10.75 | 0.38 | 11.33 | | | |
| llama.cpp-q8_0-mtp | pp2048 @ d65536 | 242.30 | 0.89 | — | 279094 | 278929 | 279094 |
| llama.cpp-q8_0-mtp | tg128 (after pp2048) @ d65536 | 10.95 | 0.37 | 11.33 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d4096 | 378.74 | 0.49 | — | 10980 | 10819 | 10980 |
| llama.cpp-q8_0-prefix-cache | pp512 @ d4096 | 355.23 | 2.26 | — | 1602 | 1441 | 1602 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp512) @ d4096 | 9.92 | 0.01 | 10.00 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d4096 | 379.72 | 1.07 | — | 10953 | 10792 | 10953 |
| llama.cpp-q8_0-prefix-cache | pp2048 @ d4096 | 366.23 | 0.14 | — | 5753 | 5592 | 5753 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp2048) @ d4096 | 9.87 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d16384 | 347.25 | 0.14 | — | 47348 | 47187 | 47348 |
| llama.cpp-q8_0-prefix-cache | pp512 @ d16384 | 289.20 | 0.44 | — | 1931 | 1770 | 1931 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp512) @ d16384 | 9.49 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d16384 | 346.63 | 0.06 | — | 47433 | 47272 | 47433 |
| llama.cpp-q8_0-prefix-cache | pp2048 @ d16384 | 305.92 | 0.38 | — | 6855 | 6695 | 6855 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp2048) @ d16384 | 9.45 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d32768 | 310.50 | 0.02 | — | 105700 | 105540 | 105700 |
| llama.cpp-q8_0-prefix-cache | pp512 @ d32768 | 236.18 | 0.68 | — | 2329 | 2168 | 2329 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp512) @ d32768 | 9.00 | 0.00 | 9.33 | | | |
| llama.cpp-q8_0-prefix-cache | ctx_pp @ d32768 | 310.04 | 0.14 | — | 105855 | 105695 | 105855 |
| llama.cpp-q8_0-prefix-cache | pp2048 @ d32768 | 254.10 | 0.27 | — | 8220 | 8060 | 8220 |
| llama.cpp-q8_0-prefix-cache | tg128 (after pp2048) @ d32768 | 8.97 | 0.00 | 9.00 | | | |
| llama.cpp-q8_0 | pp512 | 412.05 | 4.84 | — | 1407 | 1247 | 1407 |
| llama.cpp-q8_0 | tg128 (after pp512) | 10.08 | 0.00 | 11.00 | | | |
| llama.cpp-q8_0 | pp2048 | 391.48 | 0.37 | — | 5395 | 5234 | 5395 |
| llama.cpp-q8_0 | tg128 (after pp2048) | 10.02 | 0.00 | 11.00 | | | |
| llama.cpp-q8_0 | pp512 @ d4096 | 379.08 | 0.47 | — | 12319 | 12158 | 12319 |
| llama.cpp-q8_0 | tg128 (after pp512) @ d4096 | 9.90 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0 | pp2048 @ d4096 | 375.14 | 0.39 | — | 16542 | 16381 | 16542 |
| llama.cpp-q8_0 | tg128 (after pp2048) @ d4096 | 9.84 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0 | pp512 @ d16384 | 344.41 | 0.27 | — | 49220 | 49059 | 49220 |
| llama.cpp-q8_0 | tg128 (after pp512) @ d16384 | 9.46 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0 | pp2048 @ d16384 | 340.97 | 0.12 | — | 54221 | 54060 | 54221 |
| llama.cpp-q8_0 | tg128 (after pp2048) @ d16384 | 9.43 | 0.00 | 10.00 | | | |
| llama.cpp-q8_0 | pp512 @ d32768 | 308.92 | 0.11 | — | 107896 | 107736 | 107896 |
| llama.cpp-q8_0 | tg128 (after pp512) @ d32768 | 8.96 | 0.00 | 9.00 | | | |
| llama.cpp-q8_0 | pp2048 @ d32768 | 306.03 | 0.13 | — | 113930 | 113770 | 113930 |
| llama.cpp-q8_0 | tg128 (after pp2048) @ d32768 | 8.93 | 0.01 | 9.00 | | | |
| llama.cpp-q8_0 | pp512 @ d65536 | 254.76 | 0.20 | — | 259416 | 259255 | 259416 |
| llama.cpp-q8_0 | tg128 (after pp512) @ d65536 | 8.13 | 0.03 | 9.00 | | | |
| llama.cpp-q8_0 | pp2048 @ d65536 | 252.92 | 0.11 | — | 267382 | 267222 | 267382 |
| llama.cpp-q8_0 | tg128 (after pp2048) @ d65536 | 8.08 | 0.04 | 9.00 | | | |

llama-benchy 0.4.0, latency mode `generation`

## Short-chat streaming latency (256 output tokens, thinking off)

| config | TTFT s | decode t/s | TPOT ms | ITL p50 / p90 / p99 ms | e2e s |
|---|---:|---:|---:|---:|---:|
| llama.cpp-q8_0-mtp | 0.125 | 22.6 | 44.2 | 0.0 / 124.0 / 125.0 | 11.39 |
| llama.cpp-q8_0 | 0.129 | 10.1 | 99.1 | 99.0 / 99.6 / 100.8 | 25.40 |

## Coding: EvalPlus (greedy, pass@1)

| model | dataset | base | plus |
|---|---|---:|---:|
| q8_0 | humaneval | 92.1% | 89.6% |

## Power (powermetrics, CPU+GPU+ANE)

| phase | avg W | max W | GPU avg W |
|---|---:|---:|---:|
| llama-bench-q8_0 | 29.4 | 88.7 | 27.8 |
| server-q8_0-mtp | 34.7 | 73.9 | 33.4 |
| server-q8_0 | 33.6 | 73.7 | 32.3 |

