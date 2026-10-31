# H100 validation — 31 October 2026

## Accepted configuration

The final H100 PCIe run passed **125 tests with no failures or skips**, all four NVIDIA sanitizer tools on all **83 custom CUDA tests**, and **1,000 randomized layouts/content seeds × two merge implementations** against the independent NumPy FP64 oracle. Three fresh comparative reports completed with separate eager and CUDA graph measurements.

| Component | Tested value |
|---|---|
| GPU | NVIDIA H100 PCIe, 80GB, SM 9.0 |
| Driver | 595.91.07 |
| Python / compiler | 3.12.3 / GCC 13.3.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA compiler | 13.0.88 |
| Compute Sanitizer | 2026.3.0.0, build 38855100 |
| FlashInfer | 0.7.2.post1 |

## Direct evidence

| Check | Result | Raw evidence |
|---|---|---|
| Full suite | 125 passed, 0 skipped | [log](diagnostic/portable-pytest.log), [JUnit XML](diagnostic/portable-pytest.xml) |
| memcheck | 83 passed; 0 errors | [log](diagnostic/portable-memcheck.log) |
| racecheck | 83 passed; 0 errors, 0 warnings | [log](diagnostic/portable-racecheck.log) |
| synccheck | 83 passed; 0 errors | [log](diagnostic/portable-synccheck.log) |
| initcheck | 83 passed; 0 errors | [log](diagnostic/portable-initcheck.log) |
| Randomized oracle comparison | 1,000 cases; 2,000 merge executions | [source](diagnostic/stress-clean.py), [progress/result](diagnostic/stress-clean-progress.json), [log](diagnostic/portable-stress.log) |
| Tool error-detection control | Intentional out-of-bounds write detected; exit 1 expected | [source](diagnostic/negative-control.cu), [log](diagnostic/latest-negative.log) |
| Source provenance | Every tested code file SHA-256 matched published candidate | [hashes](diagnostic/final-source-sha256.json), [exact patch](diagnostic/portable-source-diff.patch) |

The maximum absolute difference in the 1,000 randomized cases was 8.091489335937396e-5. Those cases include FP16/FP32, dimensions 64/128, page sizes 16/32, MHA/GQA/MQA, empty rows, repeated pages, split boundaries and amplified logits. Acceptance tolerances were atol=rtol=0.003. This is an observed seeded result, not a universal numerical bound.

## Fresh performance results

FP16 inputs, D=128, Hq/Hkv=32/8, maximum context 4096, page size 16, split 256. Median device milliseconds below. Custom output is FP32; FlashInfer/SDPA output is FP16. FlashInfer consumes the same paged KV tensors; SDPA consumes pre-gathered/head-expanded dense KV.

| Batch | Serial eager | Shared eager | FlashInfer CUDA-core eager | Auto SDPA eager | Serial graph | Shared graph | FlashInfer graph |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.064754 | 0.056634 | 0.040299 | 0.122667 | 0.064061 | 0.054942 | 0.019687 |
| 4 | 0.174055 | 0.166086 | 0.038431 | 0.163050 | 0.171597 | 0.163792 | 0.033983 |
| 16 | 0.438090 | 0.427686 | 0.125652 | 0.572483 | 0.436432 | 0.425886 | 0.125434 |

FlashInfer CUDA-core was 1.61–4.53× faster than serial custom decode in these eager measurements. Auto SDPA dispatched to **cuDNN** on this stack. At batch 4, auto SDPA was also faster than custom serial decode. The custom implementation does not claim performance leadership. Shared merge improved these three measured shapes but remains optional because earlier T4 experiments showed regressions.

Raw reports: [batch 1](diagnostic/portable-bench-b1.json), [batch 4](diagnostic/portable-bench-b4.json), [batch 16](diagnostic/portable-bench-b16.json). Each contains thirty raw repetition means with twenty invocations per mean, randomized backend order, correctness gates and explicit unsupported-backend errors. Graph capture/compilation/planning/gathering are untimed. p95 is not serving-request tail latency.

## Compatibility investigation and retained failures

The initial H100 NVL run used PyTorch 2.8.0+cu128 and CUDA 12.8.93. All 125 correctness tests, 1,000 randomized cases and nine benchmark cases passed, but memcheck reported one CUDA API `cuKernelGetFunction` / `CUDA_ERROR_INVALID_HANDLE (400)` error on first custom launch. Eager module loading did not clear it. That run is **not** labeled sanitizer-clean. Its [original log](memcheck.log), [eager-loading retry](memcheck-eager.log), [status](status.json), nine raw benchmark JSONs and environment remain unchanged.

The same API error reproduced on H100 PCIe with Compute Sanitizer 2025.1, 2026.1.1 and 2026.3 under the CUDA 12.8/PyTorch 2.8 stack. Standalone valid CUDA, independent PyTorch extensions, and their template/stream variants passed; these probes did not establish a generic tool defect. See [old project run](diagnostic/old-custom-memcheck.log), [updated tool run](diagnostic/new-memcheck.log), [current tool run](diagnostic/latest-memcheck.log) and probe sources/logs in this directory.

Two configuration controls separate the observations: the original unmodified source passed memcheck on CUDA 13.0/PyTorch 2.11 ([log](diagnostic/original-clean-memcheck.log)); the narrow-header candidate still reported the API error on CUDA 12.8 ([log](diagnostic/portable-legacy-memcheck.log)). The validated H100 configuration is therefore the pinned CUDA 13.0/PyTorch 2.11 configuration above. The lower-level cause of the CUDA 12.8 interaction remains unproven. No API-error suppression flags were used to obtain acceptance.

The candidate replaces unused `ATen/cuda/CUDAContext.h` with the specific `c10/cuda/CUDAStream.h` include. This removes an unnecessary cuSPARSE-header build dependency; kernel arithmetic, launch geometry and public behavior are unchanged. Initial clean-stack build failures due to missing headers and FlashInfer expecting a `lib64` library directory are retained in [clean-pytest.log](diagnostic/clean-pytest.log) and [initial-portable-pytest.log](diagnostic/initial-portable-pytest.log). The library layout was corrected with a `lib64 -> lib` symlink. These are setup failures, not passing tests.

## Reproduction and provenance

The runtime checked out `2b58e7d6f7ff1221a440cc08ace7211d2c5d4406`, then applied exactly the header patch linked above. Reports correctly record tracked source as dirty. Exported hashes were independently matched to the final candidate after download; no metadata was rewritten to pretend the tree was clean. [Final environment](diagnostic/final-environment.json), [compiler version](diagnostic/clean-nvcc.log), [sanitizer version](diagnostic/latest-version.log), [artifact hashes](diagnostic/artifact-sha256.json).

Execution scripts: [isolated environment setup](diagnostic/clean_stack.py), [final acceptance](diagnostic/retest.py), [stress and benchmark runner](diagnostic/followup.py), [configuration controls](diagnostic/compare_stacks.py). NVIDIA redistribution manifests retain component URLs and checksums. Official archives were SHA-256 checked before extraction. These scripts record the actual run; the initial setup required the documented header and directory corrections before retesting.

Historical T4 results remain separate and cover their recorded source revisions. The latest header-only portability change was executed on H100; its new file hash is not retrospectively attributed to historical T4 runs. No complete language-model/server integration, request scheduler, production load qualification, end-to-end token throughput, hardware-counter profile or exhaustive correctness proof is provided.

Both GPU Pods were stopped after evidence export.
