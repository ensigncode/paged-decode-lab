# Version 0.2 engineering experiment

This candidate adds an explicit shared-memory merge option to the original
split-K paged attention implementation. The default remains `merge="serial"`.
Historical results under `results/2026-10-31-t4/` describe the original kernel;
they do not validate the new merge implementation.

## What changed

The shared merge computes each partition's softmax correction once per block,
then reuses those weights across output dimensions. The serial merge repeats
that scalar work for each dimension. Shared memory and barriers have costs, so
the new path is an experiment rather than an assumed improvement. A plan with
more than 4096 partitions falls back to serial merge; inspect
`plan.merge_implementation` to see the actual choice.

Metadata is immutable for a plan. Rebuild the plan when sequence lengths or page
mapping changes. Tests exercise ten decode lengths across page boundaries and
both merge paths, but this is not integration with an autoregressive model.

## Reproduce the comparative experiment

Use a real NVIDIA GPU with a compatible CUDA compiler and PyTorch installation.
From the repository root:

```bash
python -m pip install -e '.[gpu,test]'
python -m pip install flashinfer-python==0.7.2.post1
PYTHONPATH=src MAX_JOBS=1 python -m pytest -q
PYTHONPATH=src python benchmarks/bench_competitive.py \
  --batch 4 --context 4096 --output results/local-competitive.json
```

FlashInfer is an optional external library, licensed under Apache-2.0; its
kernels are not implemented or vendored by this project. The adapter uses
`BatchDecodeWithPagedKVCacheWrapper`, NHD layout, backend `fa2`, and the exact
same paged K/V tensors used by the custom kernel. Its output is FP16, whereas
the custom kernel outputs FP32. The reported FlashInfer workspace is the
explicitly reserved user buffer, not the library's total memory footprint.

FlashInfer CUDA-core decode on the tested T4 rejects a 32:1 query/KV head ratio.
The acceptance comparison uses supported 8:1 MQA. Our custom-kernel tests cover
other head ratios independently. Do not replace an unsupported competitor with
a different backend without recording the change.

## Measurement scope

The competitive benchmark records observed PyTorch profiler dispatch for auto
SDPA and tries forced math, flash, efficient and cuDNN backends separately.
Unsupported backends retain their errors and receive no latency result.
The primary FlashInfer CUDA-core path must run successfully. Tensor-core decode
is optional and records a failure if unsupported.

Each runner first passes an output check against FP32 eager attention. Eager
and CUDA graph replay timings are separate experiments with randomized backend
order, thirty repetitions and twenty invocations per repetition. Compilation,
planning, profiler inspection, dense gathering, and graph capture are excluded.
The dense SDPA comparison expands KV heads before timing. A percentile of
repetition means is not a request-tail latency measurement.

The suite includes independent NumPy FP64 oracle comparisons, randomized
metadata, shared pages, empty rows, large logits, graph replay, wrong-device and
strided-input rejection. Four NVIDIA Compute Sanitizer tools are required for
the new kernel before publication: memcheck, racecheck, synccheck and initcheck.
No finite suite proves complete correctness; coverage is limited to recorded
shapes, software versions, and hardware.
