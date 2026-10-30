#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q src benchmarks tools
python -m pytest -q -m 'not gpu'
