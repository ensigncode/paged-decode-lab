# Customer walkthrough

## Description before GPU acceptance

> Ensigncode is developing a CUDA laboratory for long-context LLM attention decode.
> It includes paged KV addressing, grouped-query heads, partitioned online softmax,
> a corrected merge, and a reproducible benchmark protocol. CPU reference tests
> pass; GPU validation and performance measurements are pending.

After acceptance, replace the status with the exact tested stack and link the
commit, raw evidence, and validation log. A speedup needs baseline, shape, dtype,
and measurement scope in the same sentence.

## Ten-minute demonstration

1. Explain one-token decode over a fragmented, uneven context. Establish the
   customer's real model, context distribution, GPU, and bottleneck.
2. Show `oracle.py`: independent FP64 dense math and streaming merge. Explain
   empty partitions, page boundaries, and numerical correction.
3. Walk through `decode.cu`: page lookup, lane mapping, dot reduction, recurrence,
   partial write, and merge.
4. Explain the split trade-off using the complete measured sweep, including
   regressions and uncertainty.
5. Open a real JSON report: GPU, revision, error, timing scope, raw samples, and
   wall/device differences. Explain why operator throughput is not model throughput.
6. Propose a diagnostic on the customer's workload only with a technical owner
   who has reviewed, run, and can debug the code.

The project offers inspectable CUDA indexing, numerical reasoning, warp programming,
inference memory layouts, integration, and experimental design. A repository alone
does not prove production optimization experience. Presenters should be able to
run it and explain its limits.

Useful live exercise: change a length across a page boundary, predict active pages
and partitions, and check against the oracle. Then change split size and explain
workspace growth before measuring it.

## Claims not supported

There is no measured vLLM/TensorRT-LLM comparison or customer deployment. No complete
LLM or inference server is included. Do not turn synthetic attention timings into
production cost savings, SLOs, or invented customer experience. Do not invent
reviews, contributor identities, or historical commits.

Discovery should establish prefill vs decode vs scheduling bottlenecks, actual
batch/context distributions, accuracy tolerance, and end-to-end latency requirements.
Those answers determine whether this operator is relevant.
