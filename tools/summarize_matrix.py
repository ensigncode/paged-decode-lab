#!/usr/bin/env python3
"""Summarize a complete measured matrix, including slower configurations."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "benchmark_manifest.json").read_text())
    audit = json.loads((args.directory / "source_audit.json").read_text())
    if audit["revision"] != manifest["revision"] or audit["tracked_diff"]:
        raise SystemExit("Tracked source does not match the recorded revision")
    reports = []
    for name in manifest["benchmark_reports"]:
        report = json.loads((args.directory / name).read_text())
        if report["revision"]["sha"] != manifest["revision"]:
            raise SystemExit(f"Unexpected source revision in {name}")
        if report["schema_version"] != 1 or not report["correctness"]:
            raise SystemExit(f"Missing correctness evidence in {name}")
        for result in report["results"].values():
            if result["summary"]["sample_count"] != 30 or min(result["raw"]["device_ms"]) <= 0:
                raise SystemExit(f"Unexpected timing samples in {name}")
        reports.append((name,report))
    ordinary = [(n,r) for n,r in reports if not r["configuration"]["include_gather"]]
    gather = [(n,r) for n,r in reports if r["configuration"]["include_gather"]]
    if len(ordinary) != manifest["matrix_count"] or len(gather) != manifest["additional_gather_count"]:
        raise SystemExit("Incomplete matrix")
    gpu = reports[0][1]["environment"]["gpu"]
    if any(r["environment"]["gpu"] != gpu for _,r in reports):
        raise SystemExit("Mixed hardware in one matrix")
    print(f"# Measured attention decode on {gpu}\n")
    print(f"Source revision: `{manifest['revision']}`. {len(ordinary)} matrix points and {len(gather)} supplemental gather points.\n")
    print("Each timing is median CUDA-event milliseconds over 30 repetition means, each containing 20 replays. These are static attention-operator timings, not full-model throughput or request-tail latency.\n")
    print("The benchmark dirty flag includes an untracked evidence-output directory. Tracked source was unchanged, and source_audit.json records SHA-256 hashes for every tracked file.\n")
    print("## Default split size (256 tokens)\n")
    print("| Batch | Context | Custom split ms | Single partition ms | Eager dense ms | SDPA auto dense ms | Single / split | SDPA / split |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    def timing(report, name):
        return report["results"][name]["summary"]["device_ms_p50"]
    for _,r in ordinary:
        c = r["configuration"]
        if c["split_tokens"] != 256:
            continue
        split = timing(r,"cuda_split")
        single,eager,sdpa = [timing(r,n) for n in ("cuda_single_partition","torch_eager_dense","torch_sdpa_auto_dense")]
        print(f"| {c['batch']} | {c['context']} | {split:.5f} | {single:.5f} | {eager:.5f} | {sdpa:.5f} | {single/split:.2f}x | {sdpa/split:.2f}x |")
    print("\nRatios below 1 mean the custom split path is slower. SDPA uses pre-gathered, head-expanded dense KV and an unidentified automatic backend. These results do not compare against a production paged serving engine.\n")
    print("## Complete split-size matrix\n")
    print("| Batch | Context | Split tokens | Custom ms | Single ms | Eager ms | SDPA ms | Raw evidence |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for name,r in ordinary:
        c = r["configuration"]
        values = [timing(r,n) for n in ("cuda_split","cuda_single_partition","torch_eager_dense","torch_sdpa_auto_dense")]
        formatted = " | ".join(f"{v:.5f}" for v in values)
        print(f"| {c['batch']} | {c['context']} | {c['split_tokens']} | {formatted} | [{name}]({name}) |")
    print("\n## Supplemental gather-inclusive path\n")
    print("| Batch | Context | Custom paged ms | Gather + expand + eager ms | Raw evidence |")
    print("|---:|---:|---:|---:|---|")
    for name,r in gather:
        c = r["configuration"]
        print(f"| {c['batch']} | {c['context']} | {timing(r,'cuda_split'):.5f} | {timing(r,'torch_gather_expand_eager'):.5f} | [{name}]({name}) |")
    print("\nThis final comparison includes representation conversion and allocation in the PyTorch path, while the custom path consumes paged KV directly. It does not establish an apples-to-apples kernel speedup.\n")
    print("All reports use FP16, Hq/Hkv=32/8, D=128, page size 16, seeded ragged lengths, and a single free Colab T4 session. Performance is specific to that device and environment; repeated trials on other hardware are still needed.")


if __name__ == "__main__":
    main()
