# Qwen3.8-27B on Mac mini (Apple M5 Pro, 64 GB)

Run `20261010-1516_Qwen3.8-27B_gsq-iq3s-mtp_final` · macOS 27.0 · GPU cores 20 · llama.cpp `d4d82d67f`

## Comparison

| model | size GiB | pp512 | tg128 | tg128 @ d65536 | chat tg | chat tg MTP | API tg | API tg MTP | API TTFT ms | HumanEval+ | SWE-bench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsq-iq3s-mtp | 11.29 | 400.1 | 20.0 | 13.3 | 18.3 | 28.9 | 19.3 | 24.6 | 1474 | 89.6% | — |

pp/tg in tokens/s. llama-bench columns (pp512, tg128) are the engine alone. chat tg is the short-chat stream (256 tokens, thinking off) through llama-server; API columns are llama-benchy. MTP columns are empty when MTP did not run.

- `gsq-iq3s-mtp`: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`

## llama-bench (llama.cpp, Metal)

| model | size GiB | test | t/s | ± |
|---|---:|---|---:|---:|
| gsq-iq3s-mtp | 11.28 | pp512 | 400.13 | 0.13 |
| gsq-iq3s-mtp | 11.28 | tg128 | 19.96 | 0.08 |
| gsq-iq3s-mtp | 11.28 | pp512 @ d4096 | 339.42 | 1.02 |
| gsq-iq3s-mtp | 11.28 | tg128 @ d4096 | 19.06 | 0.06 |
| gsq-iq3s-mtp | 11.28 | pp512 @ d16384 | 300.85 | 0.95 |
| gsq-iq3s-mtp | 11.28 | tg128 @ d16384 | 16.94 | 0.08 |
| gsq-iq3s-mtp | 11.28 | pp512 @ d32768 | 254.87 | 1.84 |
| gsq-iq3s-mtp | 11.28 | tg128 @ d32768 | 15.69 | 0.25 |
| gsq-iq3s-mtp | 11.28 | pp512 @ d65536 | 190.00 | 3.00 |
| gsq-iq3s-mtp | 11.28 | tg128 @ d65536 | 13.34 | 0.05 |

build `d4d82d67f`

## llama-benchy (OpenAI API, llama-bench style)

pp/tg in t/s (mean ± std); latencies in ms. TTFR = time to first response, est_ppt = estimated prompt processing time, e2e_ttft = end-to-end time to first token.

| config | test | t/s | ± | peak t/s | TTFR | est_ppt | e2e_ttft |
|---|---|---:|---:|---:|---:|---:|---:|
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 | 365.21 | 6.19 | — | 1540 | 1405 | 1540 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) | 18.74 | 0.42 | 19.33 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 (c2) | 355.23 | 2.50 | — | 2888 | 2754 | 2888 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) (c2) | 24.02 | 0.17 | 25.33 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 (c4) | 351.89 | 4.61 | — | 5831 | 5697 | 5831 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) (c4) | 40.98 | 0.04 | 44.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 | 358.52 | 6.33 | — | 5851 | 5716 | 5851 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) | 18.72 | 0.11 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 (c2) | 356.04 | 0.81 | — | 11421 | 11287 | 11421 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) (c2) | 23.21 | 0.12 | 24.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 (c4) | 344.06 | 0.47 | — | 21765 | 21631 | 21765 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) (c4) | 27.08 | 0.02 | 38.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 @ d4096 | 356.20 | 1.91 | — | 13074 | 12940 | 13074 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) @ d4096 | 18.48 | 0.05 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 @ d4096 (c2) | 347.83 | 0.48 | — | 25547 | 25412 | 25547 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) @ d4096 (c2) | 19.78 | 0.02 | 23.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp512 @ d4096 (c4) | 343.62 | 0.17 | — | 45452 | 45318 | 45452 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp512) @ d4096 (c4) | 13.57 | 0.01 | 30.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 @ d4096 | 355.34 | 0.50 | — | 17428 | 17293 | 17428 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) @ d4096 | 18.38 | 0.01 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 @ d4096 (c2) | 347.41 | 1.11 | — | 32203 | 32069 | 32203 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) @ d4096 (c2) | 14.60 | 0.01 | 20.00 | | | |
| llama.cpp-gsq-iq3s-mtp-concurrency | pp2048 @ d4096 (c4) | 344.93 | 0.13 | — | 53083 | 52948 | 53083 |
| llama.cpp-gsq-iq3s-mtp-concurrency | tg128 (after pp2048) @ d4096 (c4) | 9.23 | 0.01 | 24.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp512 | 366.28 | 2.91 | — | 1537 | 1401 | 1537 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp512) | 24.55 | 1.64 | 25.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp2048 | 361.96 | 0.96 | — | 5798 | 5662 | 5798 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp2048) | 23.07 | 1.43 | 23.33 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp512 @ d4096 | 351.09 | 0.77 | — | 13265 | 13129 | 13265 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp512) @ d4096 | 20.21 | 2.31 | 21.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp2048 @ d4096 | 347.62 | 0.12 | — | 17814 | 17677 | 17814 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp2048) @ d4096 | 18.85 | 0.73 | 19.33 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp512 @ d16384 | 323.62 | 0.44 | — | 52348 | 52212 | 52348 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp512) @ d16384 | 19.85 | 3.38 | 20.33 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp2048 @ d16384 | 319.53 | 0.74 | — | 57826 | 57690 | 57826 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp2048) @ d16384 | 17.03 | 1.03 | 17.33 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp512 @ d32768 | 290.04 | 0.11 | — | 114882 | 114746 | 114882 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp512) @ d32768 | 15.44 | 1.69 | 16.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp2048 @ d32768 | 287.50 | 0.05 | — | 121240 | 121104 | 121240 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp2048) @ d32768 | 14.64 | 1.69 | 15.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp512 @ d65536 | 240.78 | 0.15 | — | 274444 | 274308 | 274444 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp512) @ d65536 | 13.58 | 1.37 | 14.00 | | | |
| llama.cpp-gsq-iq3s-mtp-mtp | pp2048 @ d65536 | 237.66 | 0.25 | — | 284516 | 284379 | 284516 |
| llama.cpp-gsq-iq3s-mtp-mtp | tg128 (after pp2048) @ d65536 | 10.15 | 0.20 | 10.67 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d4096 | 352.65 | 2.72 | — | 11766 | 11620 | 11766 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp512 @ d4096 | 335.45 | 3.06 | — | 1673 | 1526 | 1673 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp512) @ d4096 | 18.48 | 0.06 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d4096 | 355.33 | 0.17 | — | 11679 | 11533 | 11679 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp2048 @ d4096 | 343.42 | 0.24 | — | 6110 | 5963 | 6110 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp2048) @ d4096 | 18.34 | 0.00 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d16384 | 329.78 | 1.74 | — | 49835 | 49689 | 49835 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp512 @ d16384 | 285.95 | 0.37 | — | 1937 | 1791 | 1937 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp512) @ d16384 | 17.29 | 0.01 | 18.00 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d16384 | 330.41 | 0.42 | — | 49741 | 49594 | 49741 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp2048 @ d16384 | 295.19 | 1.46 | — | 7084 | 6938 | 7084 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp2048) @ d16384 | 17.15 | 0.02 | 18.00 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d32768 | 301.65 | 1.19 | — | 108783 | 108637 | 108783 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp512 @ d32768 | 226.15 | 1.81 | — | 2410 | 2264 | 2410 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp512) @ d32768 | 15.38 | 0.10 | 16.00 | | | |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | ctx_pp @ d32768 | 301.53 | 0.07 | — | 108825 | 108679 | 108825 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | pp2048 @ d32768 | 242.41 | 1.30 | — | 8595 | 8449 | 8595 |
| llama.cpp-gsq-iq3s-mtp-prefix-cache | tg128 (after pp2048) @ d32768 | 14.81 | 0.03 | 15.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp512 | 384.77 | 6.17 | — | 1474 | 1334 | 1474 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp512) | 19.31 | 0.22 | 20.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp2048 | 368.38 | 1.10 | — | 5703 | 5562 | 5703 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp2048) | 18.90 | 0.02 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp512 @ d4096 | 359.58 | 0.61 | — | 12957 | 12817 | 12957 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp512) @ d4096 | 18.58 | 0.02 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp2048 @ d4096 | 356.12 | 0.71 | — | 17396 | 17255 | 17396 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp2048) @ d4096 | 18.40 | 0.01 | 19.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp512 @ d16384 | 333.30 | 0.20 | — | 50841 | 50701 | 50841 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp512) @ d16384 | 16.35 | 0.06 | 17.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp2048 @ d16384 | 330.42 | 0.08 | — | 55926 | 55786 | 55926 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp2048) @ d16384 | 16.38 | 0.07 | 17.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp512 @ d32768 | 300.50 | 0.15 | — | 110896 | 110756 | 110896 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp512) @ d32768 | 14.82 | 0.01 | 15.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp2048 @ d32768 | 297.24 | 0.58 | — | 117277 | 117137 | 117277 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp2048) @ d32768 | 14.81 | 0.07 | 15.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp512 @ d65536 | 249.72 | 0.49 | — | 264630 | 264490 | 264630 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp512) @ d65536 | 12.68 | 0.02 | 13.00 | | | |
| llama.cpp-gsq-iq3s-mtp | pp2048 @ d65536 | 247.83 | 0.26 | — | 272844 | 272703 | 272844 |
| llama.cpp-gsq-iq3s-mtp | tg128 (after pp2048) @ d65536 | 12.63 | 0.02 | 13.00 | | | |

llama-benchy 0.4.0, latency mode `generation`

## Short-chat streaming latency (256 output tokens, thinking off)

| config | TTFT s | decode t/s | TPOT ms | ITL p50 / p90 / p99 ms | e2e s |
|---|---:|---:|---:|---:|---:|
| llama.cpp-gsq-iq3s-mtp-mtp | 0.101 | 28.9 | 34.6 | 0.1 / 96.5 / 96.8 | 8.93 |
| llama.cpp-gsq-iq3s-mtp | 0.105 | 18.3 | 54.6 | 54.9 / 55.3 / 55.5 | 14.02 |

## Coding: EvalPlus (greedy, pass@1)

| model | dataset | base | plus |
|---|---|---:|---:|
| gsq-iq3s-mtp | humaneval | 93.3% | 89.6% |

## Power (powermetrics, CPU+GPU+ANE)

| phase | avg W | max W | GPU avg W |
|---|---:|---:|---:|
| llama-bench-gsq-iq3s-mtp | 35.6 | 78.1 | 33.8 |
| server-gsq-iq3s-mtp-mtp | 36.0 | 76.0 | 35.2 |
| server-gsq-iq3s-mtp | 35.4 | 73.5 | 34.4 |

