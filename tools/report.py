#!/usr/bin/env python3
"""Render an actual benchmark JSON as a Markdown comparison for review."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    with open(args.report) as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1 or not report.get("correctness"):
        raise SystemExit("Expected a schema v1 report with correctness evidence")
    print(f"GPU: {report['environment']['gpu']} | revision: {report['revision']['sha']}")
    print("\nOperator timings only. p95 describes repetition means, not serving tail latency.\n")
    print("| Backend | Device p50 (ms) | Device p95 (ms) | Wall p50 (ms) |")
    print("|---|---:|---:|---:|")
    for name, result in report["results"].items():
        value = result["summary"]
        print(f"| {name} | {value['device_ms_p50']:.5f} | {value['device_ms_p95']:.5f} | {value['wall_ms_p50']:.5f} |")
    print("\nSDPA auto-selected its backend. Dense preparation is excluded except in the gather row.")


if __name__ == "__main__":
    main()
