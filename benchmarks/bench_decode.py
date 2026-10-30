#!/usr/bin/env python3
"""Benchmark one static decode shape; writes measured evidence, never examples."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import torch
from paged_decode import __version__
from paged_decode.metrics import summarize
from paged_decode.runtime import DecodePlan
from paged_decode.workload import make_metadata, make_tensors
from paged_decode.reference import gather_dense, expand_gqa, eager_attention, sdpa_attention


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--dim", type=int, choices=[64,128], default=128)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--page-size", type=int, choices=[16,32], default=16)
    p.add_argument("--split-tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--inner", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--include-gather", action="store_true",
                   help="also time paged gather + GQA expansion + eager attention")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.repetitions, a.inner, a.warmup) <= 0:
        p.error("repetitions, inner and warmup must be positive")
    if a.context > 1048576:
        p.error("context must not exceed 1M tokens")
    return a


def revision():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def driver_snapshot():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader"], text=True, timeout=5).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def measure(fn, device, inner):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / inner
    return start.elapsed_time(end) / inner, wall_ms


def main():
    a = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("NVIDIA GPU unavailable; no benchmark results were produced")
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    dtype = getattr(torch, a.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    layout, table, lengths = make_metadata(a.batch, a.heads, a.kv_heads, a.dim,
        a.context, a.page_size, seed=a.seed, ragged=not a.uniform)
    q, k, v = make_tensors(layout, device=device, dtype=dtype, seed=a.seed)
    plan = DecodePlan(layout, table, lengths, device=device, split_tokens=a.split_tokens)
    single = DecodePlan(layout, table, lengths, device=device, split_tokens=a.context)
    dense_k, dense_v, mask = gather_dense(k, v, table, lengths)
    expanded_k, expanded_v = expand_gqa(q, dense_k, dense_v)
    runners = {
        "cuda_split": lambda: plan.run(q,k,v),
        "cuda_single_partition": lambda: single.run(q,k,v),
        "torch_eager_dense": lambda: eager_attention(q,expanded_k,expanded_v,mask),
        "torch_sdpa_auto_dense": lambda: sdpa_attention(q,expanded_k,expanded_v,mask),
    }
    if a.include_gather:
        def full_path():
            dk, dv, valid = gather_dense(k,v,table,lengths)
            ek, ev = expand_gqa(q,dk,dv)
            return eager_attention(q,ek,ev,valid)
        runners["torch_gather_expand_eager"] = full_path
    with torch.inference_mode():
        reference = eager_attention(q,expanded_k,expanded_v,mask)
        errors = {}
        # A failed correctness gate prevents writing performance results.
        for name, fn in runners.items():
            output = fn().float()
            torch.testing.assert_close(output, reference, atol=3e-3, rtol=3e-3)
            errors[name] = {"max_abs_vs_fp32_eager": (output-reference).abs().max().item(),
                           "atol": 3e-3, "rtol": 3e-3}
        for fn in runners.values():
            for _ in range(a.warmup):
                fn()
        torch.cuda.synchronize(device)
        memory = {"allocated_bytes": torch.cuda.memory_allocated(device),
                  "reserved_bytes": torch.cuda.memory_reserved(device)}
        raw = {name: {"device_ms": [], "wall_ms": []} for name in runners}
        order = []
        rng = random.Random(a.seed)
        before = driver_snapshot()
        for _ in range(a.repetitions):
            names = list(runners)
            rng.shuffle(names)
            order.append(names)
            for name in names:
                device_ms, wall_ms = measure(runners[name], device, a.inner)
                raw[name]["device_ms"].append(device_ms)
                raw[name]["wall_ms"].append(wall_ms)
        after = driver_snapshot()
    props = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "package_version": __version__, "revision": revision(),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key,value in vars(a).items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "gpu": props.name, "compute_capability": [props.major, props.minor],
            "gpu_memory_bytes": props.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi_before": before, "nvidia_smi_after": after},
        "workload": {"lengths": lengths.tolist(), "physical_pages": layout.physical_pages,
            "cache_bytes": (k.numel()+v.numel()) * k.element_size(),
            "expanded_dense_kv_bytes": (expanded_k.numel()+expanded_v.numel()) * k.element_size(),
            "split_workspace_bytes": plan.workspace_bytes,
            "single_workspace_bytes": single.workspace_bytes,
            "steady_state_memory": memory},
        "correctness": errors,
        "measurement": {"timing": "CUDA events and synchronized wall clock",
            "sample_unit": "mean per replay within one repetition; p95 is NOT request-tail latency",
            "scope": "static attention decode operator; excludes model and serving scheduler",
            "sdpa_backend": "automatic, not identified; no FlashAttention claim",
            "dense_baselines": "pre-gathered, GQA-expanded KV; excludes preparation unless named gather",
            "sequence_tokens_per_second": "batch / operator latency; NOT full-model generation throughput",
            "inner_replays": a.inner, "warmup_replays": a.warmup,
            "trial_order": order},
        "results": {name: {"summary": summarize(samples["device_ms"], samples["wall_ms"], a.batch),
                           "raw": samples} for name,samples in raw.items()},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {a.output}; all correctness gates passed")
    for name, result in report["results"].items():
        print(f"{name:30s} {result['summary']['device_ms_p50']:.5f} ms median")


if __name__ == "__main__":
    main()
