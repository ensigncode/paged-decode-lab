#!/usr/bin/env bash
# Run from the repository root on an existing NVIDIA GPU host.
set -euo pipefail
python -c 'import torch; assert torch.cuda.is_available(), "GPU required"'
command -v nvcc
command -v compute-sanitizer
python -m pytest -q tests/test_oracle.py tests/test_metrics.py tests/test_cuda.py
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py -k 'test_ragged_decode' --maxfail=1
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py -k 'test_nondefault_stream' --maxfail=1
