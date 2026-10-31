#!/usr/bin/env bash
# Run from the repository root on an existing NVIDIA GPU host.
set -euo pipefail
python -c 'import torch; assert torch.cuda.is_available(), "GPU required"'
command -v nvcc
command -v compute-sanitizer
python -m pytest -q --maxfail=1
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py --maxfail=1
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py --maxfail=1
compute-sanitizer --tool synccheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py --maxfail=1
compute-sanitizer --tool initcheck --error-exitcode 1 \
  python -m pytest -q tests/test_cuda.py --maxfail=1
