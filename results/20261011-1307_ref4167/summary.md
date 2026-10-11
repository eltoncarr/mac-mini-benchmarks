# Qwen3.8-27B on Mac mini (Apple M5 Pro, 64 GB)

Run `20261011-1307_ref4167` · macOS 27.0 · GPU cores 20 · llama.cpp `d4d82d67f`

## Llama-2 7B reference (llama.cpp Discussion #4167 format)

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 7B F16                   |  12.55 GiB |     6.74 B | MTL,BLAS   |       6 |           pp512 |       1589.81 ± 1.69 |
| llama 7B F16                   |  12.55 GiB |     6.74 B | MTL,BLAS   |       6 |           tg128 |         21.45 ± 0.01 |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | MTL,BLAS   |       6 |           pp512 |       1579.50 ± 0.94 |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | MTL,BLAS   |       6 |           tg128 |         38.97 ± 0.35 |
| llama 7B Q4_0                  |   3.53 GiB |     6.74 B | MTL,BLAS   |       6 |           pp512 |       1645.99 ± 1.44 |
| llama 7B Q4_0                  |   3.53 GiB |     6.74 B | MTL,BLAS   |       6 |           tg128 |         66.84 ± 0.15 |

build: c1d0e7a00 (10621)

Summary row for the thread's table (bandwidth from Apple's spec for the chip):

| Chip | BW (GB/s) | GPU Cores | F16 PP | F16 TG | Q8_0 PP | Q8_0 TG | Q4_0 PP | Q4_0 TG | commit |
|---|---|---|---|---|---|---|---|---|---|
| M5 Pro (Mac mini, 64 GB) | 307 | 20 | 1589.81 | 21.45 | 1579.50 | 38.97 | 1645.99 | 66.84 | c1d0e7a00 |

