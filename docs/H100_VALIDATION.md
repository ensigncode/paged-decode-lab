# H100 acceptance and reproduction

The [complete H100 record](../results/2026-10-31-h100/SUMMARY.md) links every
passing check, initial failure, control experiment, raw sample and source hash.
The accepted configuration is H100 PCIe 80GB, driver 595.91.07, CUDA compiler
13.0.88, PyTorch 2.11.0+cu130, Python 3.12.3, FlashInfer 0.7.2.post1 and
Compute Sanitizer 2026.3.0.0 (build 38855100).

## Existing compatible GPU host

A real H100, compatible driver, C++17 compiler, full CUDA 13.0 development toolkit,
and the recorded sanitizer release must already be available. The following
commands install packages and run checks; they do not rent or purchase compute.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[test]' ninja flashinfer-python==0.7.2.post1
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
export PATH="$CUDA_HOME/bin:$PATH"
# Ensure compute-sanitizer resolves to the recorded 2026.3 release.
compute-sanitizer --version
nvcc --version
bash tools/gpu_acceptance.sh
PYTHONPATH=src python benchmarks/bench_competitive.py \
  --batch 4 --context 4096 --output results/my-h100-b4.json
```

For the 1,000-case oracle comparison, run the published
[stress script](../results/2026-10-31-h100/diagnostic/stress-clean.py) after
changing only its progress-output path to a writable local location. Its fixed
seed, input distribution and comparison tolerances are inspectable. Preserve
any change in your run's source audit.

The actual experiment assembled the CUDA compiler components from NVIDIA's
checksum-verified redistribution archives into an isolated prefix. The
[setup script](../results/2026-10-31-h100/diagnostic/clean_stack.py), manifests,
[final runner](../results/2026-10-31-h100/diagnostic/retest.py) and
[final environment](../results/2026-10-31-h100/diagnostic/final-environment.json) preserve
that execution. CUDA's `lib64` location must resolve to the actual library
directory for FlashInfer; the experiment used `lib64 -> lib`. Initial setup
failures are retained rather than retroactively edited out of the logs.

## Source change and compatibility scope

The only CUDA source change is replacing the broad ATen CUDA context header
with the specific c10 stream header. This removes an unused cuSPARSE-header
build dependency. No arithmetic, addressing, reduction, launch geometry or API
semantics changed. All tested code hashes match the published candidate.

On the H100 CUDA 12.8/PyTorch 2.8 configuration, memcheck reports a first-launch
CUDA API invalid-handle error even though correctness tests pass. Both original
and narrow-header sources were checked; updating the sanitizer alone did not
clear the report. The original source passes memcheck on the accepted CUDA 13.0
stack. These observations establish a tested configuration, not the precise
lower-level cause of the older stack's interaction. Do not suppress the report
or label that older configuration sanitizer-clean.

A deliberate invalid-write control was detected by the accepted sanitizer.
Full acceptance covers the published finite suite and distributions. Model
integration, serving load, hardware-counter profiling and exhaustive correctness
are outside this record. T4 evidence continues to describe its original hashes.

The same final kernel source also passed the [fresh T4 acceptance and profiling run](../results/2026-10-31-final-t4/SUMMARY.md). This adds current T4 evidence without changing the historical records or attributing T4 counters to H100.
