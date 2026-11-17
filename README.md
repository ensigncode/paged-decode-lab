# Paged Decode Lab

**Ensigncode Software · CUDA kernels, inference workloads, and benchmark methodology**

A clean-room CUDA C++ implementation of **split-K paged grouped-query attention (GQA)** for single-token LLM decode. Built as an open verification and profiling benchmark harness: it traverses non-contiguous KV pages directly, maintains an online softmax in registers, and merges split-K partitions without materializing an attention-score matrix.

> **Need custom H100 / H200 / Blackwell (B200) Tensor-Core kernels, FP8 KV-cache quantization, or an Nsight Compute bottleneck audit?**  
> See [Commercial H200 & Production Kernel Engineering](#commercial-h200--production-kernel-engineering) or [schedule a technical review](https://ensigncode.com).

---

## Verification & Hardware Telemetry Summary

Unlike unverified demo kernels, every build in this repository passes strict hardware-level memory/race sanitization and numerical oracle gates before emitting performance telemetry:

| Verification Gate | Environment / Scope | Measured Status |
| :--- | :--- | :--- |
| **NVIDIA H100 PCIe Validation** | CUDA 13.0.88 · PyTorch 2.11.0+cu130 · Driver 580.82.07 | **125 / 125 tests passed** (0 skips) |
| **NVIDIA Sanitizer Suite** | `memcheck`, `racecheck`, `initcheck`, `synccheck` across 83 CUDA tests | **0 errors · 0 racecheck warnings** |
| **Numerical Parity Gate** | 1,000 randomized ragged/shared-page cases × both merge kernels | **100% pass** vs. independent FP64 NumPy oracle |
| **CUDA Graph Replay** | Preallocated workspace on caller's CUDA stream | **Zero-allocation replay** outside graph capture |
| **T4 Profiling & Trace Suite** | 36-config sweep + 9 CUDA activity traces + Nsight Compute (`ncu`) | **Complete** (split-size & stage breakdown) |

### Baseline Positioning (Transparent Engineering)
* **What this public baseline measures:** A scalar-FMA, warp-shuffle CUDA C++ reference kernel (`FP32` accumulation/output) validated on NVIDIA H100 PCIe and Tesla T4.
* **Comparison against FlashInfer:** In H100 eager comparisons, FlashInfer's Tensor-Core/vectorized CUDA decode is **1.61–4.53× faster** than this scalar baseline (`FP16` vs. `FP32` output).
* **Why we publish it:** This repository demonstrates our **kernel engineering discipline, sanitizer-clean C++ bindings, CUDA Graph compatibility, and reproducible benchmarking methodology**. For production engagements, we extend these architectures with **Hopper TMA (`cp.async.bulk`), Tensor Core (`wgmma`) tiling, inter-head GQA KV reuse, and FP8/FP4 quantized KV caches**.

---

## Technical Architecture

| Component | Specification |
| :--- | :--- |
| **Core Kernel** | Custom CUDA C++ (`src/paged_decode/csrc/decode.cu`) with PyTorch C++ extension bindings |
| **Attention Layouts** | Single-token decode supporting **MHA, GQA, and MQA** head ratios |
| **Paged KV Memory** | Physical layout `[physical_page, token, kv_head, head_dim]` indexed via logical page table |
| **Threadblock & Warp Design** | 4 warp states per CTA; split-K multi-CTA parallelization per query head for small-batch SM occupancy |
| **Online Softmax & Merge** | Register-resident online max/denominator; max-corrected serial merge + optional bounded shared-memory merge |
| **Supported Shapes** | `FP16` / `FP32` inputs · `head_dim` 64 / 128 · `page_size` 16 / 32 · ragged lengths, empty contexts, shared pages |
| **Benchmark Harness** | Automated correctness gate, raw latency samples, device/wall timings, and source-hash metadata |

---

## Quickstart & Reproduction

### 1. CPU Oracle & Metrics Verification (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests/test_oracle.py tests/test_metrics.py
```

### 2. GPU Kernel Compilation & Full Acceptance Suite
Requires Linux, an NVIDIA GPU, C++17 compiler, CUDA toolkit (`nvcc`), and CUDA-enabled PyTorch (`>=2.6,<3`):
```bash
python -m pip install ninja
python -m pytest -q tests/test_cuda.py
bash tools/gpu_acceptance.sh
```
*(Note: Importing `paged_decode` does not trigger `torch` imports or JIT compilation; kernel compilation occurs explicitly during `DecodePlan` initialization outside timed regions.)*

### 3. Run Single Workload or 36-Configuration Hardware Sweep
```bash
# Single GQA decode benchmark (Llama-3-8B head geometry, 4K context)
python benchmarks/bench_decode.py \
  --batch 4 --heads 32 --kv-heads 8 --dim 128 \
  --context 4096 --page-size 16 --split-tokens 256 \
  --include-gather --output results/local-4k.json

python tools/report.py results/local-4k.json

# Full 36-configuration hardware sweep
bash tools/sweep.sh results/my-gpu
```

---

## Python API Example

```python
from paged_decode.runtime import DecodePlan
from paged_decode.workload import make_metadata, make_tensors

# Configure batch=4, 32 Q heads, 8 KV heads (GQA 4:1), dim=128, context=4096, page_size=16
layout, pages, lengths = make_metadata(4, 32, 8, 128, 4096, 16)
q, k, v = make_tensors(layout, device="cuda:0")

# Explicit plan compilation & workspace preallocation (CUDA-graph safe)
plan = DecodePlan(layout, pages, lengths, device="cuda:0", split_tokens=256)
output = plan.run(q, k, v)  # Returns FP32 tensor [batch, query_heads, head_dim]
```

---

## Evidence, Raw Traces & Documentation

* **H100 Raw Evidence & Reports:** [results/2026-10-31-h100/SUMMARY.md](results/2026-10-31-h100/SUMMARY.md) · [H100 Reproduction Guide](docs/H100_VALIDATION.md)
* **T4 Acceptance, Nsight Compute & Traces:** [Final T4 Run](results/2026-10-31-final-t4/SUMMARY.md) · [Hardening v0.2](results/2026-10-31-hardening/SUMMARY.md) · [Sweep v0.1](results/2026-10-31-t4/SUMMARY.md)
* **Architecture & Methodology:**
  * [Kernel Design & Split-K Merge Equations](docs/DESIGN.md)
  * [Benchmark Protocol & Fairness Controls](docs/BENCHMARKING.md)
  * [Validation & Sanitizer Methodology](docs/VALIDATION.md)
  * [Customer Engineering Walkthrough](docs/PORTFOLIO.md)
  * [Profiling Findings & Optimization Roadmap](docs/PROFILING.md)

---

## Commercial H200 & Production Kernel Engineering

This public repository serves as our open-source verification lab. For enterprise AI teams running production inference clusters on **NVIDIA H100, H200, or Blackwell B200**, **Ensigncode Software** delivers custom kernel and runtime engineering under NDA:

* **Hopper / H200 HBM3e Optimization:** Warp-specialized kernels utilizing Tensor Memory Accelerator (`TMA`), asynchronous pipelines, and inter-head GQA KV reuse to saturate H200's `4.8 TB/s` memory bandwidth.
* **Quantized KV-Cache & Low-Precision Kernels:** Custom `FP8` (`E4M3`/`E5M2`) and `FP4` dequantization fused directly into attention decode and GEMM epilogues.
* **5-Day Nsight GPU Bottleneck Audit:** Deep kernel-level profiling (`Nsight Systems` + `Nsight Compute` warp-stall, memory-coalescing, and roofline diagnostics) with a fixed-bid implementation roadmap.

**Contact our GPU Engineering Team:** [ensigncode.com](https://ensigncode.com) · [Customer Walkthrough](docs/PORTFOLIO.md)

---

## References & License

* Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
* Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)
* Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180)
* Ainslie et al., [Grouped-query attention](https://arxiv.org/abs/2305.13245)

Released under the [MIT License](LICENSE).
