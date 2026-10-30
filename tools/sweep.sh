#!/usr/bin/env bash
# Runs on the current host only; does not provision infrastructure.
set -euo pipefail
output_dir="${1:?Usage: bash tools/sweep.sh results/unique-run-name}"
mkdir -p "$output_dir"
for batch in 1 4 16; do
  for context in 1024 4096 16384; do
    for split in 128 256 512 1024; do
      python benchmarks/bench_decode.py --batch "$batch" --context "$context" \
        --split-tokens "$split" \
        --output "$output_dir/b${batch}-l${context}-s${split}.json"
    done
  done
done
