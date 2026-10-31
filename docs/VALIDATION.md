# Historical version 0.1 NVIDIA GPU validation record

Current H100 acceptance: [H100 validation](H100_VALIDATION.md).

Completed 31 October 2026 (India time), using a free Google Colab Tesla T4 runtime.
No paid compute, credits, hardware purchase, or subscription upgrade was used.

## Executed configuration

| Component | Observed value |
|---|---|
| GPU | NVIDIA Tesla T4, 15360 MiB, compute capability 7.5 |
| Driver | 580.82.07 |
| CUDA toolkit | 13.0.88 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.13.16 |
| Host compiler | GCC 13.3.0 |
| Tested source revision | `b120cb28f879a8a215a2626399aa89454e51fda8` |

This record covers version 0.1 at the revision above. Version 0.2 changes the
CUDA merge and validation code; see [hardening evidence](../results/2026-10-31-hardening/SUMMARY.md)
for its separate execution record. Do not attribute these historical results to
the new implementation.

## Results and direct evidence

| Check | Observed result | Evidence |
|---|---|---|
| Full test suite, GPU enabled | **57 passed**, no skipped tests | [pytest.log](../results/2026-10-31-t4/pytest.log), [JUnit XML](../results/2026-10-31-t4/pytest.xml) |
| GPU tests under memcheck | 28 passed; **0 errors** | [memcheck.log](../results/2026-10-31-t4/memcheck.log) |
| Nondefault stream/graph test under racecheck | 1 passed; **0 errors, 0 warnings** | [racecheck.log](../results/2026-10-31-t4/racecheck.log) |
| GPU tests under synccheck | 28 passed; **0 errors** | [synccheck.log](../results/2026-10-31-t4/synccheck.log) |
| Benchmark matrix | **36 points completed**, all correctness gates passed | [manifest](../results/2026-10-31-t4/benchmark_manifest.json), [complete summary](../results/2026-10-31-t4/SUMMARY.md) |
| Supplemental gather-inclusive measurements | **6 points completed**, all correctness gates passed | Same manifest and summary |
| Source provenance | Every tracked file's SHA-256 matched the recorded Git commit | [source_audit.json](../results/2026-10-31-t4/source_audit.json) |
| Environment | Actual driver/toolkit/device output retained | [environment.json](../results/2026-10-31-t4/environment.json) |
| Separate CUDA 12.4 build check | Compilation, device code generation for sm_75/sm_80, linking and module import passed | [build report](../results/2026-10-31-t4/cuda12-build.json), [build log](../results/2026-10-31-t4/cuda12-build.log) |

The 57 tests comprise 29 CPU/reference checks and 28 CUDA tests. GPU coverage
includes FP16/FP32, dimensions 64/128, page sizes 16/32, split boundaries,
ragged/empty contexts, shuffled and shared pages, GQA/MQA/MHA, long-context tails,
large finite logits, invalid inputs, the current stream, and CUDA graph replay.

The benchmark custom kernels' maximum absolute difference from the FP32 eager
reference across the 42 reports was 2.384185791015625e-7. This is a measured error
for seeded synthetic inputs, not a universal accuracy bound. All 42 reports retain
30 device/wall timing samples per backend, with 20 replays inside each sample.

The general benchmark dirty flag includes the untracked evidence output directory.
Tracked source had no changes; its recorded hashes were independently compared
against the tested Git commit after export. No result metadata was rewritten to
pretend that the working tree was wholly clean.

## Reproduce

On an existing NVIDIA GPU host with compatible PyTorch, CUDA toolkit, and Compute
Sanitizer installed, run from the repository root:

```bash
python -m pip install -e '.[test]' ninja
bash tools/gpu_acceptance.sh
bash tools/sweep.sh results/unique-run-name
```

The acceptance script refuses to proceed when a real CUDA device is unavailable.
The default sweep measures the 36 operator-only points. Supplemental measurements
use `benchmarks/bench_decode.py --include-gather` at batch 1/4, context 1024/4096/16384,
and split size 256. Unique output paths are required.

Build-only hosts can use `tools/build_extension.py --architectures '7.5;8.0'
--output results/build.json` with a complete matching CUDA toolchain and CUDA-enabled
PyTorch. Such a result is explicitly marked as build-only; it is not GPU acceptance.

## Practical limits

This is a validated experimental attention operator on one device/software stack.
It is not a complete model server or a certified production integration. SDPA uses
pre-gathered, head-expanded dense KV with automatic backend selection; no tuned
paged-engine comparison or FlashAttention claim is made. The p95 statistics are
percentiles of repetition means, not per-request serving latency. Other hardware,
real model workloads, and hardware-counter profiling remain future validation work.
