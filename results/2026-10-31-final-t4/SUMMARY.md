# Final-revision T4 validation and profiling — 31 October 2026

The exact published revision `41a672407c7185c8d5c86bb9dc50ef69031c3478`, including the narrow CUDA stream header, passed a fresh free-Colab Tesla T4 run. All 17 recorded source hashes match the published source. No kernel or default tuning parameter changed in this follow-up.

## Acceptance

| Check | Observed result | Evidence |
|---|---|---|
| Full suite | 125 passed; 0 failures, errors or skips | [log](pytest.log), [JUnit](pytest.xml) |
| memcheck | 83 passed; zero errors | [log](memcheck.log) |
| racecheck | 83 passed; zero errors or warnings | [log](racecheck.log) |
| synccheck | 83 passed; zero errors | [log](synccheck.log) |
| initcheck | 83 passed; zero errors | [log](initcheck.log) |
| Randomized FP64 oracle | 1,000 cases; 2,000 merge executions; maximum absolute error 8.091489335937396e-5 | [progress](stress-progress.json), [script](stress.py) |
| Competitive cases | B1 and B4, context 4096, eager and CUDA graphs; all correctness gates passed | [B1](competitive-b1.json), [B4](competitive-b4.json) |

Environment: Tesla T4 15,360 MiB, driver 580.82.07, Python 3.13.16, PyTorch 2.11.0+cu130, nvcc 13.0.88, FlashInfer 0.7.2.post1, Compute Sanitizer 2025.3.1.0. See [environment](environment.log), [compiler](nvcc.log), [GPU](nvidia-smi.log), [package freeze](freeze.log), [source hashes](source-sha256.json), [executed acceptance cell](acceptance-runner.py), [status](FINISHED.json).

In the competitive run, median eager microseconds for serial/shared/FlashInfer CUDA-core were 214.148/206.980/91.725 at B1 and 575.919/578.551/218.283 at B4. Automatic SDPA selected the efficient backend. Unsupported forced backends are retained explicitly in the reports. Shared merge regressed slightly at B4; no default was promoted.

## CUDA stage attribution

The separate [profiling driver](profiling/profile-driver.py) captured nine complete CUDA activity traces: serial, shared and FlashInfer at B1/B4/B16. Each trace contains ten invocations. All trace summaries contain device kernel events, and all profiling error dictionaries are empty. [Full matrix and summaries](profiling/profiles/results.json).

Fraction of summed custom serial device-kernel time in these traces:

| Batch | Attention partial stage | Merge stage |
|---:|---:|---:|
| 1 | 93.2% | 6.8% |
| 4 | 96.1% | 3.9% |
| 16 | 97.6% | 2.4% |

The partial stage dominates these measured T4 shapes. Removing the merge alone cannot close the observed FlashInfer gap. This is stage attribution, not a claim that profiled durations equal steady-state latency. The traces show substantial timing variation even for the identical partial kernel across serial/shared captures, so a difference between those captures must not be attributed to merge code.

## Nsight Compute counters

Nsight Compute 2025.3.1.0 successfully profiled the partial and serial merge kernels at B4, context 4096, Hq/Hkv=32/8, D128, page 16, split 256. Nineteen replay passes were used per kernel. [Original binary report](profiling/ncu-report.ncu-rep), [raw metric CSV](profiling/ncu-raw.csv), [readable detail CSV](profiling/ncu-details.csv), [tool log](profiling/ncu.log), [version](profiling/ncu-version.log).

| Metric | Partial stage | Serial merge |
|---|---:|---:|
| Achieved occupancy | 91.21% | 38.02% |
| DRAM throughput, percentage of peak | 28.98% | 42.75% |
| Compute/SM throughput | 65.70% | 46.41% |
| Registers per thread | 39 | 36 |
| Local memory spilling requests | 0 | 0 |
| Waves per SM | 6.40 | 0.40 |

The report records an SM frequency near 585 MHz. This shared free runtime does not provide controlled application clocks; do not compare these durations directly with earlier runs or dedicated hardware.

These counters do not support labeling the partial stage simply DRAM-bandwidth-saturated or low-occupancy. Nsight's balance rule identifies both computation and memory traffic as optimization targets. Source inspection shows scalar online recurrence, warp reductions per token, and separate logical KV reads for query heads sharing a KV head. Grouped-head/tiled reuse is a next experiment, not an established cause or promised speedup; cache reuse means logical duplicate loads are not equal to duplicate DRAM traffic.

## Split-size exploration

The matrix checks all five split sizes 64/128/256/512/1024 with both merges and one same-layout FlashInfer CUDA-core baseline at B1/B4/B16, context 4096. All 33 runners passed correctness gates; all 33 CUDA graphs captured and passed the same checks. Each timing mode retains 30 randomized-order trials with 20 invocations per trial.

The best custom eager median was only about 3–4% below default serial split-256 in this exploration. The winning split/merge differs by batch and timing mode. These are selected tuning observations without an independent holdout run or clock control, not a robust optimization claim. FlashInfer remains faster. Defaults and kernel code are unchanged.

## Provenance and limits

The [executed profiling cell](profiling/runner.py) records the actual queued source, and the driver has its own SHA-256 inside the report. The runtime's acceptance ZIP was captured before profiling appended command status. This publication restores the original acceptance snapshot byte-for-byte and places subsequent status and artifacts under `profiling/`; no test result or raw trace was rewritten. The original 24-entry acceptance manifest and 10-entry trace manifest verify independently; the [publication manifest](manifest-sha256.json) covers both collections.

H100 correctness and benchmark acceptance remain documented [separately](../2026-10-31-h100/SUMMARY.md). H100 stage/counter profiling has not yet been collected. T4 measurements do not establish H100 bottlenecks. No exhaustive correctness, production readiness or performance-leadership claim is made. The free runtime was released after export and hash verification.
