# Validation record

Prepared 31 October 2026 (India time).

| Check | Evidence/status |
|---|---|
| Python source compilation | Passed with Python 3.12 |
| Independent CPU numerical/metadata/metrics tests | 23 passed |
| Actual PyTorch baseline vs NumPy integration | 6 passed, CPU PyTorch 2.6.0+cpu |
| Full available test run | 29 passed; CUDA module skipped because NVIDIA GPU is absent |
| Benchmark CLI help | Passed |
| Missing-GPU benchmark behavior | Exits with an explanatory error; produces no result JSON |
| Bash syntax | Checked |
| CUDA compilation | Not run: `nvcc` unavailable |
| GPU correctness tests | Written, not executed |
| Nondefault stream and graph replay | Written, not executed |
| Compute Sanitizer | Acceptance commands provided, not executed |
| GPU profiling and performance | Not measured |

CPU numerical tests exercise split sizes 1/17/256, dimensions 64/128, shuffled
pages, zero/one/multiple tokens, large scores, shared prefixes, and empty merges.
CPU PyTorch tests exercise MQA/GQA/MHA and all-empty/ragged batches against an
independent FP64 oracle. They do not compile or execute CUDA.

The GPU suite includes FP16/FP32, both head dimensions and page sizes, unusual split
boundaries, empty rows, MQA and MHA mappings, a long-context tail, large logits,
repeated pages, invalid inputs, current-stream behavior, and graph replay. These
are acceptance specifications until run on a real GPU.

Reproduce CPU checks:

```bash
python -m pip install -e '.[test]'
bash tools/cpu_check.sh
# Optional actual torch baseline checks need a PyTorch installation:
python -m pytest -q tests/test_reference.py
```

Before treating this project as measured portfolio evidence, run
`bash tools/gpu_acceptance.sh` on an existing compatible NVIDIA host. Retain its
complete output, exact torch/toolkit/compiler/driver versions, device model, and
commit. Then run the benchmark matrix, publish raw results, and describe failures
and regressions. A skipped test, successful import, or CPU algebra check is not
GPU acceptance. No paid infrastructure or hosted runner has been started.
