#!/usr/bin/env bash
# Existing, already-authorized GPU host only. Does not provision or stop a rental.
# A command timeout limits the process, not a cloud provider's GPU billing.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
output=$1
if [[ -e "$output" ]]; then
  echo 'Refusing to overwrite existing evidence' >&2
  exit 2
fi
mkdir -p "$output"
python -c 'import torch; assert torch.cuda.is_available(), "A real NVIDIA GPU is required"'
nvidia-smi > "$output/nvidia-smi.log"
nvcc --version > "$output/nvcc.log"
python -m pip freeze > "$output/freeze.log"
timeout 20m python benchmarks/profile_decode.py --output "$output/matrix" \
  > "$output/profile.log" 2>&1
if command -v ncu >/dev/null 2>&1; then
  ncu --version > "$output/ncu-version.log"
  set +e
  timeout 5m ncu --nvtx --nvtx-include 'profile/' --set detailed \
    --launch-count 2 --export "$output/ncu-report" \
    python benchmarks/profile_decode.py --ncu-target --output "$output/ncu-target" \
    > "$output/ncu.log" 2>&1
  ncu_status=$?
  set -e
  printf '%s\n' "$ncu_status" > "$output/ncu-exit-code.txt"
else
  printf '%s\n' 'ncu not found in PATH; hardware counters were not collected.' \
    > "$output/ncu-unavailable.txt"
fi
python - "$output" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}
(root/'artifact-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
