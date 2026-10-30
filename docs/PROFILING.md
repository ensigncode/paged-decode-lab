# Profiling experiments

No profiler findings have been collected. Run acceptance first on an existing GPU.

| Hypothesis | Experiment | What would reject it |
|---|---|---|
| Small batch/long context underoccupies the GPU | Split-size sweep at matched shape | More CTAs do not reduce time, or merge dominates |
| GQA causes redundant traffic | Hq=32,Hkv=1/8/32; inspect memory counters | L2 reuse hides most extra reads |
| Excessive splits hurt merge | Inspect partial vs merge duration | Partial stage still dominates |
| Scalar recurrence is instruction-bound | Inspect exp/shuffle stalls and bandwidth | DRAM is already saturated |
| Ragged grid wastes scheduling | Matched live-token count, different length skew | Empty CTAs have negligible cost |

Example commands on a host with Nsight already installed:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=results/decode-trace \
  python benchmarks/bench_decode.py --repetitions 3 --inner 5 \
  --output results/profile-run.json
ncu --set full --kernel-name-base demangled \
  --kernel-name 'regex:.*(partial_decode|merge_decode).*' \
  --launch-skip 10 --launch-count 4 --export results/decode-kernels \
  python benchmarks/bench_decode.py --repetitions 1 --inner 1 \
  --output results/ncu-run.json
```

Available metrics and CLI behavior depend on tool/GPU versions. Verify selected
kernel names/shapes: correctness and warmup launches affect launch-skip. Profiled
runs have substantial overhead and are not normal timing results.

After measurement, consider KV tile reuse across GQA heads, tiled rather than
one-score-at-a-time recurrence, hierarchical merging, or compact ragged work queues.
Each change needs correctness tests and matched before/after JSON, including
regressions. Native serving-engine integration needs a separate API contract and
competitive engine baseline. Do not assume this experimental layout is drop-in.
