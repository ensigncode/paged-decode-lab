# Measured attention decode on Tesla T4

Source revision: `b120cb28f879a8a215a2626399aa89454e51fda8`. 36 matrix points and 6 supplemental gather points.

Each timing is median CUDA-event milliseconds over 30 repetition means, each containing 20 replays. These are static attention-operator timings, not full-model throughput or request-tail latency.

The benchmark dirty flag includes an untracked evidence-output directory. Tracked source was unchanged, and source_audit.json records SHA-256 hashes for every tracked file.

## Default split size (256 tokens)

| Batch | Context | Custom split ms | Single partition ms | Eager dense ms | SDPA auto dense ms | Single / split | SDPA / split |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 0.08986 | 0.25513 | 0.39956 | 0.10770 | 2.84x | 1.20x |
| 1 | 4096 | 0.18115 | 1.12272 | 1.51270 | 0.33307 | 6.20x | 1.84x |
| 1 | 16384 | 0.71930 | 4.46104 | 5.89942 | 1.40620 | 6.20x | 1.95x |
| 4 | 1024 | 0.19207 | 0.33999 | 1.48318 | 0.30360 | 1.77x | 1.58x |
| 4 | 4096 | 0.51760 | 1.39773 | 5.85489 | 1.12178 | 2.70x | 2.17x |
| 4 | 16384 | 2.52218 | 5.43725 | 23.20463 | 4.64095 | 2.16x | 1.84x |
| 16 | 1024 | 0.54181 | 0.63042 | 5.97257 | 1.09855 | 1.16x | 2.03x |
| 16 | 4096 | 2.88114 | 3.18498 | 23.61104 | 4.49574 | 1.11x | 1.56x |
| 16 | 16384 | 11.34898 | 12.45972 | 92.59252 | 17.50572 | 1.10x | 1.54x |

Ratios below 1 mean the custom split path is slower. SDPA uses pre-gathered, head-expanded dense KV and an unidentified automatic backend. These results do not compare against a production paged serving engine.

## Complete split-size matrix

| Batch | Context | Split tokens | Custom ms | Single ms | Eager ms | SDPA ms | Raw evidence |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1024 | 128 | 0.06492 | 0.23326 | 0.39676 | 0.11885 | [b1-l1024-s128.json](b1-l1024-s128.json) |
| 1 | 1024 | 256 | 0.08986 | 0.25513 | 0.39956 | 0.10770 | [b1-l1024-s256.json](b1-l1024-s256.json) |
| 1 | 1024 | 512 | 0.13027 | 0.22113 | 0.39509 | 0.09800 | [b1-l1024-s512.json](b1-l1024-s512.json) |
| 1 | 1024 | 1024 | 0.24094 | 0.22465 | 0.39830 | 0.10500 | [b1-l1024-s1024.json](b1-l1024-s1024.json) |
| 1 | 4096 | 128 | 0.19422 | 1.12051 | 1.51320 | 0.33224 | [b1-l4096-s128.json](b1-l4096-s128.json) |
| 1 | 4096 | 256 | 0.18115 | 1.12272 | 1.51270 | 0.33307 | [b1-l4096-s256.json](b1-l4096-s256.json) |
| 1 | 4096 | 512 | 0.17695 | 1.14556 | 1.51233 | 0.33623 | [b1-l4096-s512.json](b1-l4096-s512.json) |
| 1 | 4096 | 1024 | 0.30411 | 1.13342 | 1.51317 | 0.33038 | [b1-l4096-s1024.json](b1-l4096-s1024.json) |
| 1 | 16384 | 128 | 0.76073 | 4.42776 | 5.89882 | 1.37426 | [b1-l16384-s128.json](b1-l16384-s128.json) |
| 1 | 16384 | 256 | 0.71930 | 4.46104 | 5.89942 | 1.40620 | [b1-l16384-s256.json](b1-l16384-s256.json) |
| 1 | 16384 | 512 | 0.75304 | 4.47819 | 5.88993 | 1.43525 | [b1-l16384-s512.json](b1-l16384-s512.json) |
| 1 | 16384 | 1024 | 0.75915 | 4.53191 | 5.88950 | 1.46277 | [b1-l16384-s1024.json](b1-l16384-s1024.json) |
| 4 | 1024 | 128 | 0.16847 | 0.34108 | 1.48377 | 0.30505 | [b4-l1024-s128.json](b4-l1024-s128.json) |
| 4 | 1024 | 256 | 0.19207 | 0.33999 | 1.48318 | 0.30360 | [b4-l1024-s256.json](b4-l1024-s256.json) |
| 4 | 1024 | 512 | 0.18826 | 0.34000 | 1.48331 | 0.30353 | [b4-l1024-s512.json](b4-l1024-s512.json) |
| 4 | 1024 | 1024 | 0.32823 | 0.33032 | 1.48466 | 0.30034 | [b4-l1024-s1024.json](b4-l1024-s1024.json) |
| 4 | 4096 | 128 | 0.53139 | 1.37728 | 5.85651 | 1.11491 | [b4-l4096-s128.json](b4-l4096-s128.json) |
| 4 | 4096 | 256 | 0.51760 | 1.39773 | 5.85489 | 1.12178 | [b4-l4096-s256.json](b4-l4096-s256.json) |
| 4 | 4096 | 512 | 0.56982 | 1.38819 | 5.85189 | 1.12507 | [b4-l4096-s512.json](b4-l4096-s512.json) |
| 4 | 4096 | 1024 | 0.72711 | 1.39579 | 5.85022 | 1.13075 | [b4-l4096-s1024.json](b4-l4096-s1024.json) |
| 4 | 16384 | 128 | 2.76819 | 5.54967 | 23.35612 | 4.72398 | [b4-l16384-s128.json](b4-l16384-s128.json) |
| 4 | 16384 | 256 | 2.52218 | 5.43725 | 23.20463 | 4.64095 | [b4-l16384-s256.json](b4-l16384-s256.json) |
| 4 | 16384 | 512 | 2.45039 | 5.42580 | 23.20100 | 4.62151 | [b4-l16384-s512.json](b4-l16384-s512.json) |
| 4 | 16384 | 1024 | 2.49534 | 5.46188 | 23.20023 | 4.64582 | [b4-l16384-s1024.json](b4-l16384-s1024.json) |
| 16 | 1024 | 128 | 0.56843 | 0.65079 | 5.98344 | 1.10839 | [b16-l1024-s128.json](b16-l1024-s128.json) |
| 16 | 1024 | 256 | 0.54181 | 0.63042 | 5.97257 | 1.09855 | [b16-l1024-s256.json](b16-l1024-s256.json) |
| 16 | 1024 | 512 | 0.56766 | 0.63664 | 5.96939 | 1.10000 | [b16-l1024-s512.json](b16-l1024-s512.json) |
| 16 | 1024 | 1024 | 0.61382 | 0.64352 | 5.94546 | 1.09920 | [b16-l1024-s1024.json](b16-l1024-s1024.json) |
| 16 | 4096 | 128 | 3.09933 | 3.26967 | 23.75486 | 4.51198 | [b16-l4096-s128.json](b16-l4096-s128.json) |
| 16 | 4096 | 256 | 2.88114 | 3.18498 | 23.61104 | 4.49574 | [b16-l4096-s256.json](b16-l4096-s256.json) |
| 16 | 4096 | 512 | 2.78092 | 3.15948 | 23.59103 | 4.43247 | [b16-l4096-s512.json](b16-l4096-s512.json) |
| 16 | 4096 | 1024 | 2.81110 | 3.17594 | 23.59440 | 4.44839 | [b16-l4096-s1024.json](b16-l4096-s1024.json) |
| 16 | 16384 | 128 | 11.93621 | 12.38632 | 92.58452 | 17.55048 | [b16-l16384-s128.json](b16-l16384-s128.json) |
| 16 | 16384 | 256 | 11.34898 | 12.45972 | 92.59252 | 17.50572 | [b16-l16384-s256.json](b16-l16384-s256.json) |
| 16 | 16384 | 512 | 10.99282 | 12.38580 | 92.57729 | 17.51540 | [b16-l16384-s512.json](b16-l16384-s512.json) |
| 16 | 16384 | 1024 | 10.98091 | 12.43183 | 92.58367 | 17.52313 | [b16-l16384-s1024.json](b16-l16384-s1024.json) |

## Supplemental gather-inclusive path

| Batch | Context | Custom paged ms | Gather + expand + eager ms | Raw evidence |
|---:|---:|---:|---:|---|
| 1 | 1024 | 0.08955 | 0.76870 | [gather-b1-l1024-s256.json](gather-b1-l1024-s256.json) |
| 1 | 4096 | 0.20260 | 2.72230 | [gather-b1-l4096-s256.json](gather-b1-l4096-s256.json) |
| 1 | 16384 | 0.78899 | 11.63261 | [gather-b1-l16384-s256.json](gather-b1-l16384-s256.json) |
| 4 | 1024 | 0.19528 | 2.94690 | [gather-b4-l1024-s256.json](gather-b4-l1024-s256.json) |
| 4 | 4096 | 0.54528 | 10.09323 | [gather-b4-l4096-s256.json](gather-b4-l4096-s256.json) |
| 4 | 16384 | 2.52536 | 43.24855 | [gather-b4-l16384-s256.json](gather-b4-l16384-s256.json) |

This final comparison includes representation conversion and allocation in the PyTorch path, while the custom path consumes paged KV directly. It does not establish an apples-to-apples kernel speedup.

All reports use FP16, Hq/Hkv=32/8, D=128, page size 16, seeded ragged lengths, and a single free Colab T4 session. Performance is specific to that device and environment; repeated trials on other hardware are still needed.
