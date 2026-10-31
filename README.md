# Paged Decode Lab

**Ensigncode Software · CUDA kernels, inference workloads, and benchmark methodology**

A CUDA C++ implementation of **split-K paged grouped-query attention (GQA)** for
the single-token decode step of an LLM. It follows fragmented KV pages directly,
keeps an online softmax in registers, and merges independent context partitions
without materializing an attention-score tensor.

**Version 0.2 validated on a real NVIDIA Tesla T4; experimental operator study.**
All **125 tests passed**: 39 CPU/reference checks, 83 custom CUDA tests, and
3 external-library comparisons. Each of NVIDIA memcheck, racecheck, synccheck,
and initcheck ran all 83 custom CUDA tests with zero reported errors; racecheck
also reported zero warnings.

The new **10-case competitive matrix** covers MHA, GQA, MQA, batches 1/4,
contexts up to 16K, and separate eager/CUDA-graph measurements. FlashInfer uses
the same paged KV tensors. PyTorch profiler identified automatic SDPA as the
efficient backend. Forced unsupported backends are recorded explicitly.
See the [current results and raw evidence](results/2026-10-31-hardening/SUMMARY.md)
and [reproduction details](docs/HARDENING.md).

**Measured outcome:** FlashInfer CUDA-core decode was **1.22–2.88× faster** than
our serial custom kernel in these eager measurements. The optional shared merge
had mixed results; the default remains serial. Custom output is FP32, while
FlashInfer output is FP16. This portfolio demonstrates CUDA implementation,
numerical testing, and honest experimental analysis; it makes no performance
leadership or production-serving claim.

The [historical 42-case split-size sweep](results/2026-10-31-t4/SUMMARY.md)
is retained with its [version 0.1 validation record](docs/VALIDATION.md).
Its dense-reference speedups do not establish an advantage over a tuned paged engine.

## Why this problem

During generation, each new query attends over a growing KV cache. A long context
with a small active batch can provide too few independent thread blocks to occupy
a large GPU. Serving workloads also have uneven lengths and fragmented pages.
Splitting the context exposes more work, but adds a merge and intermediate traffic.
The right split size is an experiment, not a universal constant.

This is an operator laboratory. It does not load weights, schedule requests, or
generate text. It provides a concrete problem to inspect, measure, and optimize.

## Implementation

| Component | Details |
|---|---|
| Kernel | CUDA C++ with PyTorch extension binding |
| Attention | One query token per sequence; MHA, GQA, MQA head mappings |
| Cache | `[physical_page, token, kv_head, dimension]` plus logical page table |
| Parallelism | Four warp states per CTA; multiple CTAs per query head |
| Numerics | Online maximum/denominator; FP32 accumulation and output |
| Merge | Max-corrected reduction; serial default, optional bounded shared-weight merge |
| Inputs | FP16/FP32; head dimensions 64/128; page sizes 16/32 |
| Edge cases | Ragged lengths, empty contexts, partial and shared pages |
| Replay | Preallocated workspace; caller's current CUDA stream |
| References | Independent FP64 NumPy oracle; PyTorch eager/identified SDPA; same-layout FlashInfer |
| Benchmark | Correctness gate, raw samples, device/wall timings, environment metadata |

This version uses scalar FMA and warp shuffles. It has no Tensor Cores, async
copies, quantized KV, or inter-head KV reuse. GQA heads sharing one KV head still
issue separate reads. The measured FlashInfer gap is published above. Further optimization needs
profiling and new measurements.

## Setup

CPU review needs Python 3.10+, NumPy, and pytest:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests/test_oracle.py tests/test_metrics.py
```

GPU execution additionally needs Linux, an NVIDIA GPU, a C++17 compiler, a local
CUDA toolkit containing `nvcc`, and a compatible CUDA-enabled PyTorch installation.
Install PyTorch according to its official instructions for your driver/toolkit.
A CUDA-enabled wheel alone does not supply the full extension compiler toolchain.
GPU validation used PyTorch 2.11.0+cu130, CUDA 13.0.88, driver 580.82.07,
and Python 3.13.16 on a Tesla T4. A separate historical version 0.1 GPU-less build check passed with
PyTorch 2.6.0+cu124 and CUDA 12.4.131 for sm_75 and sm_80 code generation;
that second check does not establish execution on another GPU.

```bash
# After installing a suitable CUDA-enabled torch>=2.6,<3:
python -m pip install ninja
python -m pytest -q tests/test_cuda.py
bash tools/gpu_acceptance.sh
```

Importing `paged_decode` does not import torch or compile CUDA. Compilation is an
explicit part of plan preparation, outside benchmarks and intended graph capture.

## Measured experiment

```bash
python benchmarks/bench_decode.py \
  --batch 4 --heads 32 --kv-heads 8 --dim 128 \
  --context 4096 --page-size 16 --split-tokens 256 \
  --include-gather --output results/local-4k.json
python tools/report.py results/local-4k.json
bash tools/sweep.sh results/my-gpu
```

All paths must pass a numerical comparison before a performance report is written.
Reports never silently overwrite existing files. The sweep runs 36 configurations.
No GPU rentals, paid services, or hosted CI are configured.

## API

```python
from paged_decode.runtime import DecodePlan
from paged_decode.workload import make_metadata, make_tensors

layout, pages, lengths = make_metadata(4, 32, 8, 128, 4096, 16)
q, k, v = make_tensors(layout, device="cuda:0")
plan = DecodePlan(layout, pages, lengths, device="cuda:0", split_tokens=256)
output = plan.run(q, k, v)  # FP32 [batch, query_heads, head_dim]
# Borrowed storage: clone output if it must survive another run.
```

Metadata is immutable per plan; rebuild for changed lengths or page assignments.
KV contents may change. Separate plans are needed for concurrent streams. Inputs
must be finite, contiguous, on the plan device, and detached from autograd. Huge
finite inputs can still overflow FP32 dot products; arbitrary-precision behavior
is not supported.

## Read the code and decisions

- [Kernel design and merge equations](docs/DESIGN.md)
- [Benchmark protocol and fairness](docs/BENCHMARKING.md)
- [Validation evidence](docs/VALIDATION.md)
- [Customer walkthrough](docs/PORTFOLIO.md)
- [Profiling hypotheses](docs/PROFILING.md)
- [Contributing](CONTRIBUTING.md)

Core kernel: `src/paged_decode/csrc/decode.cu`. Replay API: `runtime.py`.
Independent CPU reference: `oracle.py`.

## Limits and next work

No prefill, cache allocator/eviction, RoPE, KV writes, ALiBi, sliding-window mask,
dropout, backward pass, tensor parallelism, or serving-engine integration is
implemented. The cache is read-only during an invocation. The query is assumed
to be at the end of the live cache, so every live token is visible.

The T4 correctness and sanitizer acceptance checks are complete. Next work is
hardware-counter profiling and broader hardware/workload coverage to decide between
GQA reuse, hierarchical merge, tiled attention, and a compact ragged work grid.

## Algorithm lineage

Online softmax and paged KV attention are established techniques. This is a new
compact implementation for experimentation, not a claim to invent the algorithms
or reproduce a production engine. No external kernel code is vendored.

- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180).
- Ainslie et al., [Grouped-query attention](https://arxiv.org/abs/2305.13245).

MIT licensed; see [LICENSE](LICENSE).
