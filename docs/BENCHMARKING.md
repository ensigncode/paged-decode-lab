# Benchmark protocols

Version 0.2 uses `benchmarks/bench_competitive.py` (schema 2): same-paged-layout
FlashInfer, observed SDPA dispatch, forced-backend support results, and separate
eager and CUDA graph measurements. See [hardening methodology](HARDENING.md) and
[current evidence](../results/2026-10-31-hardening/SUMMARY.md).

The remainder of this document describes the historical version 0.1
`bench_decode.py` protocol (schema 1). Its backend-identification and graph-timing
limitations apply to those historical reports.

# Historical schema 1 protocol

The [Tesla T4 evidence](../results/2026-10-31-t4/SUMMARY.md) contains 36 matrix
points and six supplemental gather-inclusive points, each gated on correctness.
They were measured in one free Colab session. No rented GPU or paid service was used.

## Paths and scope

| Row | Timed work | Excluded preparation |
|---|---|---|
| cuda_split | Paged partial + merge kernels | Compile, metadata upload/checks, allocation |
| cuda_single_partition | Same algorithm, one context partition | Same preparation |
| torch_eager_dense | FP32 matmul, mask, softmax, weighted sum | Paged gather and GQA expansion |
| torch_sdpa_auto_dense | Automatic PyTorch SDPA | Paged gather and GQA expansion |
| torch_gather_expand_eager | Gather, expansion, eager attention and allocations | Original paged-cache creation |

The single-partition ablation still uses four token-processing warps and a merge.
SDPA uses a different layout; this is not a comparison with vLLM or TensorRT-LLM
paged kernels. The auto-selected SDPA backend is not identified and is never
labeled FlashAttention. GQA head expansion is explicit and its memory is reported.

## Correctness gate

Small GPU tests compare to independent FP64 NumPy attention. The benchmark gates
all paths against FP32 PyTorch eager attention with TF32 disabled. A failed
`assert_close` stops before writing performance evidence. Benchmark atol/rtol are
0.003; ordinary-logit GPU tests use tighter tolerances. Keep actual errors.

Coverage includes shuffled pages, GQA/MQA, empty contexts, partially filled pages,
large logits, shared prefixes, split sizes including one token, and merge identities.
The documented T4 run executed all 57 tests, including 28 GPU tests.
Coverage on another device or software stack still requires a new acceptance run.

## Sampling and interpretation

Compile and allocate before timing. Warm every backend. Randomize backend order
per repetition with a recorded seed. Synchronize before CUDA events around `inner`
replays, wait for the end event, and collect synchronized wall time. Divide each
repetition's duration by `inner`; retain every raw sample. Report median, minimum,
and interpolated p95 over those repetition means.

**p95 here is not per-request serving tail latency.** Device events may include
idle gaps caused by host dispatch. Wall timing exposes additional host overhead.
Graph replay timing is not implemented; graph behavior has a separate test.
Fixed inputs and warmup create a steady-state replay workload, not a production
traffic simulation.

`sequence_tokens_per_second = B*1000/device_ms` describes one attention operator.
It excludes projections, MLPs, weights, sampling, and scheduling; it is not model
generation throughput and cannot directly establish customer cost savings.

The allocator snapshot includes all resident backends; it is not per-backend peak
memory. Use isolated processes for peak-memory experiments. Logical KV bytes are
not measured HBM traffic: head reuse and hardware caches change physical reads.
Use profiler counters for bandwidth claims.

## Shape matrix

The sweep runs B=1/4/16, context=1K/4K/16K, split=128/256/512/1024 at Hq/Hkv=32/8,
D=128, FP16, page size 16: 36 configurations, excluding gather preparation.
The measured evidence also includes six gather-inclusive points at B=1/4.
Use --include-gather for an individual supplemental run. Extend to D=64, page size 32, FP32,
MQA, and uniform contexts. Expanded dense references can require substantial VRAM.
An OOM is an experiment failure; document a reduced shape, not an incomplete
comparison disguised as a successful full run. Use unique output directories.

Compare matched shapes, tolerances, distributions, and timing scopes. Publish
regressions. Do not cherry-pick a tuned split and call it an untuned default.
Separate tuning and evaluation workloads when selecting parameters.

## Evidence

JSON records raw device/wall samples, seed, lengths, commit and dirty state, GPU
model/capability, torch/CUDA build, driver snapshot, shape, dtype, warmup/inner
counts, randomized order, tolerance, and correctness errors. Before/after GPU
snapshots include temperature, power, and clocks when available; this is not
continuous thermal monitoring or proof of an isolated GPU.

Record concurrent workloads, MIG, power policy, toolkit/compiler versions, and
profiler findings separately. Never invent unavailable environment details.
