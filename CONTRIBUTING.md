# Contributing

Tie each change to a falsifiable hypothesis. Explain indexing and numerical
invariants in comments; avoid speculative specialization without measurements.

Correctness changes need a regression test. Performance changes need matched
before/after reports, hardware/settings, raw samples, shape coverage, and regressions.
Do not loosen tolerance solely to make an optimization pass without explaining it.

Run CPU tests for every change. Kernel/stream/workspace changes require GPU tests
and compute-sanitizer before acceptance. A skipped test is not a pass. Keep the
NumPy oracle independent of kernel indexing helpers so shared mistakes are less
likely. No paid/hosted runner is configured.
