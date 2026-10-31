# Profiling findings and experiments

The [fresh T4 record](../results/2026-10-31-final-t4/SUMMARY.md) includes
nine complete CUDA activity traces, a real Nsight Compute report with readable
counter CSVs, and a five-split/two-merge exploration at batches 1/4/16.
The source matches the final H100-validated kernel. No kernel or default changed.

## Measured T4 findings

- The attention partial stage accounts for about 93–98% of summed custom serial
  kernel time in the three traces. Optimizing only the merge cannot close the
  measured FlashInfer gap.
- At B4/L4096, Nsight Compute measured partial-stage occupancy of 91.21%, zero
  local-memory spill requests and DRAM throughput at 28.98% of peak. The partial
  stage cannot be described simply as low-occupancy or DRAM-bandwidth-saturated.
- The serial merge has only 0.40 waves per SM at this shape, but its small share
  of total kernel time limits the opportunity from merge-only changes.
- Split tuning changed best observed eager medians by only about 3–4%. Winners
  vary by batch and timing mode. Shared merge still has regressions. There is no
  independently confirmed tuning win and no default change.

These are T4 findings. **H100 stage/counter profiling has not been collected.**
The [H100 acceptance record](../results/2026-10-31-h100/SUMMARY.md) contains
correctness, sanitizer and matched timing evidence, not hardware-counter claims.

## Reproduce on an existing compatible GPU

Use the pinned stack for the GPU being tested. Plans, compilation, profiling and
graph capture are outside the benchmark sample regions. The profiling driver
checks correctness before timing every runner and checks graph outputs again.

```bash
PYTHONPATH=src python benchmarks/profile_decode.py --output results/my-profile
# Optional orchestration: environment capture, bounded processes and NCU attempt.
PYTHONPATH=src bash tools/profile_existing_gpu.sh results/my-profile-batch
```

The Python driver was executed on T4. The shell orchestration is syntax-checked;
it does not rent a GPU or stop cloud billing. A process timeout is not a provider
shutdown policy. Capture the output before releasing an ephemeral runtime.

For two targeted Nsight Compute launches after untimed warmup:

```bash
PYTHONPATH=src ncu --nvtx --nvtx-include 'profile/' --set detailed \
  --launch-count 2 --export results/my-ncu \
  python benchmarks/profile_decode.py --ncu-target --output results/my-ncu-target
ncu --import results/my-ncu.ncu-rep --page details --csv > results/my-ncu-details.csv
ncu --import results/my-ncu.ncu-rep --page raw --csv > results/my-ncu-raw.csv
```

The `profile` push/pop range contains one partial-stage launch and one serial
merge launch at B4/L4096. The NCU mode does not run the timing matrix. Keep tool
errors visible; do not weaken host security settings or silently suppress
permission errors. See NVIDIA's [CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [performance-counter permission explanation](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Interpretation and next work

Chrome traces provide CUDA activity durations, not hardware performance counters.
The Nsight report provides the separately replayed counter experiment. Profiler
instrumentation, replay and clock behavior perturb duration. Do not mix these
numbers with unprofiled medians, infer request tail latency from repetition means,
or transfer a T4 bottleneck conclusion to H100.

The T4 report records about 585 MHz SM frequency. The free shared runtime did not
provide fixed application clocks, and identical partial kernels varied across
captures. The raw trial order, samples, trace events and counter report remain
available so these limitations can be inspected.

Source inspection motivates grouped-head KV reuse and tiled score/softmax work
as next experiments. The existing GQA kernel issues separate logical KV reads
per query head; cache hits mean this is not equal to duplicated DRAM traffic.
Any optimization needs independent correctness/sanitizer acceptance and matched
before/after measurements, including regressions, on its target GPU.
