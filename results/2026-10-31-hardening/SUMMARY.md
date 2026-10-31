# Version 0.2: Tesla T4 hardening evidence

Completed in a free Colab T4 session. This is a synthetic attention-operator study, not a complete inference server.

## Acceptance

Full suite: **125 passed**, comprising 39 CPU/reference checks, 83 custom CUDA tests, and 3 FlashInfer comparisons. No skips in this run. Each NVIDIA sanitizer ran all 83 custom CUDA tests: memcheck, racecheck, synccheck and initcheck. All reported zero errors; racecheck also reported zero warnings.

Raw evidence: [pytest](pytest-revised.log), [JUnit](pytest-revised.xml), [memcheck](memcheck.log), [racecheck](racecheck.log), [synccheck](synccheck.log), [initcheck](initcheck.log), [environment](environment.json).

The first run stopped after 87 passes when FlashInfer rejected MQA group size 32 on the T4. Its [failure log](pytest-initial-failed.log) and XML are retained. The revised external MQA test uses group size 8. Custom-kernel MQA coverage is independent of that competitor restriction.

## Competitive measurements

Ten shapes, thirty randomized repetitions, twenty invocations per repetition, with correctness gates before timing. Each JSON includes raw device and wall samples, eager and CUDA graph results, unsupported-backend errors, actual SDPA dispatch and source hashes. All auto SDPA dispatches below were identified through PyTorch profiler operators; no GPU bandwidth claims are made.

Latencies below are median device milliseconds. FlashInfer is the CUDA-core path; native FP16 output versus FP32 custom output. Both consume the same paged NHD KV tensors. SDPA uses pre-gathered, head-expanded dense KV. Planning, compilation, profiler inspection, capture and gathering are excluded.

| Shape (B/L/D/Hq/Hkv/page) | Serial eager | Shared eager | FlashInfer eager | Serial graph | Shared graph | FlashInfer graph | SDPA auto backend |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/1024/128/32/8/16 | 0.093311 | 0.092561 | 0.076435 | 0.085892 | 0.084922 | 0.065804 | efficient |
| 1/16384/128/32/8/16 | 0.799951 | 0.761583 | 0.298297 | 0.791756 | 0.762562 | 0.287379 | efficient |
| 1/4096/128/32/8/16 | 0.210387 | 0.202930 | 0.089688 | 0.209888 | 0.204708 | 0.087686 | efficient |
| 1/4096/128/8/1/16 | 0.067070 | 0.062795 | 0.035761 | 0.066221 | 0.060996 | 0.027794 | efficient |
| 1/4096/64/8/8/32 | 0.087159 | 0.082639 | 0.054722 | 0.087830 | 0.083894 | 0.049708 | efficient |
| 4/1024/128/32/8/16 | 0.200000 | 0.199379 | 0.090847 | 0.201346 | 0.204969 | 0.091198 | efficient |
| 4/16384/128/32/8/16 | 2.827404 | 3.055215 | 1.010475 | 2.899306 | 2.868655 | 0.887723 | efficient |
| 4/4096/128/32/8/16 | 0.582395 | 0.591946 | 0.232849 | 0.590020 | 0.598898 | 0.218618 | efficient |
| 4/4096/128/8/1/16 | 0.189997 | 0.185034 | 0.066010 | 0.192182 | 0.186928 | 0.062831 | efficient |
| 4/4096/64/8/8/32 | 0.202761 | 0.197006 | 0.118550 | 0.202397 | 0.196147 | 0.115067 | efficient |

FlashInfer CUDA-core was 1.22–2.88× faster than serial custom merge in eager measurements. Serial/shared eager latency ratio was 0.925–1.068; shared is optional and the default remains serial. These are shape-specific observations, not a hardware-wide guarantee.

Maximum observed absolute difference versus FP32 eager across all recorded runners was 0.000140145421. All passed atol=rtol=0.003. This is not a universal error bound.

## Provenance and limits

The runtime checked out candidate `67feff86753c6efc2a7eabe1c0a239046d7dd5fd`, then applied the single external-test MQA shape edit shown in [source-diff.patch](source-diff.patch). Reports correctly retain `tracked_code_dirty=true`. Every source/test/benchmark file was exported with [SHA-256 hashes](source-sha256.json) and independently matched to the published candidate. The kernel itself was not edited during acceptance. Publication commits differ because files were uploaded through GitHub.

FlashInfer version 0.7.2.post1, PyTorch 2.11.0+cu130, CUDA 13.0.88, driver 580.82.07, Python 3.13.16. The installation log retains a resolver conflict with unused preinstalled numba-cuda after cuda-core was upgraded; this project does not import Numba. This was an ephemeral Colab environment, not a globally dependency-clean production image.

T4 only. No second-GPU execution, complete model integration, request scheduler, concurrent serving load, end-to-end generation throughput, GPU hardware-counter profile, production deployment or exhaustive correctness proof. Historical version 0.1 results remain separate. Percentiles describe repetition means, not request-tail latency. External workspace bytes count the explicitly reserved buffer, not total library memory.

The project demonstrates indexing, numerical reasoning, implementation, testing and transparent measurement. The measured tuned-library gap is an optimization opportunity, not a claim of performance leadership.
