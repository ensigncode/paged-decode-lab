#!/usr/bin/env python3
"""Attribute decode GPU time and test split tuning without changing kernels.

Chrome traces are profiling evidence, never used as steady-state timings.
The tuning matrix uses separately sampled eager and CUDA graph measurements.
No hardware-counter conclusion is inferred from CUDA activity durations.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from paged_decode.competitors import FlashInferPlan
from paged_decode.reference import eager_attention, expand_gqa, gather_dense
from paged_decode.runtime import DecodePlan
from paged_decode.workload import make_metadata, make_tensors
from bench_competitive import sample, source_record


def trace_runner(fn, name, output, launches=10):
    """Retain the complete CPU/CUDA trace and summarize only device kernels."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as trace:
        with torch.profiler.record_function(name):
            for _ in range(launches):
                fn()
        torch.cuda.synchronize()
    path = output / (name + '.trace.json')
    trace.export_chrome_trace(str(path))
    record = json.loads(path.read_text())
    kernels = {}
    for event in record['traceEvents']:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X':
            continue
        row = kernels.setdefault(event['name'], {'calls': 0, 'total_us': 0.0})
        row['calls'] += 1
        row['total_us'] += event['dur']
    if not kernels:
        raise RuntimeError('CUDA trace contains no kernel activities')
    for row in kernels.values():
        row['mean_us'] = row['total_us'] / row['calls']
    return {'trace': path.name, 'invocations': launches, 'kernels': kernels,
            'scope': 'CUDA activity tracing; timings include profiler perturbation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--inner', type=int, default=20)
    parser.add_argument('--ncu-target', action='store_true')
    args = parser.parse_args()
    if min(args.context, args.repetitions, args.inner) <= 0:
        parser.error('context and sample counts must be positive')
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing evidence')
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    environment = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0),
                   'driver': subprocess.check_output(['nvidia-smi'], text=True)}
    result = {'source': source_record(), 'environment': environment,
              'profile_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'context': args.context, 'repetitions': args.repetitions,
              'inner': args.inner, 'batches': {}, 'scope':
              'Static FP16 paged attention; custom FP32 output, FlashInfer FP16. '
              'No planning, compilation or profiling in timed samples. '
              'CUDA activity traces are not hardware performance counters.'}
    for batch in ([4] if args.ncu_target else [1, 4, 16]):
        layout, table, lengths = make_metadata(batch, 32, 8, 128, args.context, 16)
        q, k, v = make_tensors(layout)
        dk, dv, mask = gather_dense(k, v, table, lengths)
        ek, ev = expand_gqa(q, dk, dv)
        with torch.inference_mode():
            reference = eager_attention(q, ek, ev, mask)
            plans, runners = {}, {}
            for split in ([256] if args.ncu_target else [64, 128, 256, 512, 1024]):
                for merge in ['serial', 'shared']:
                    name = f'custom_s{split}_{merge}'
                    plan = DecodePlan(layout, table, lengths, split_tokens=split, merge=merge)
                    plans[name] = plan
                    runners[name] = lambda plan=plan: plan.run(q, k, v)
            if args.ncu_target:
                fn = runners['custom_s256_serial']
                for _ in range(10): fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push('profile')
                fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                (args.output / 'target.json').write_text(json.dumps(result, indent=2))
                return
            fi = FlashInferPlan(layout, table, lengths, device=q.device)
            runners['flashinfer_cuda_core'] = lambda: fi.run(q, k, v)
            correctness = {}
            for name, fn in runners.items():
                output = fn().float()
                torch.testing.assert_close(output, reference, atol=3e-3, rtol=3e-3)
                correctness[name] = {'max_abs': (output-reference).abs().max().item(),
                                     'atol': 3e-3, 'rtol': 3e-3}
                for _ in range(10): fn()
            torch.cuda.synchronize()
            eager, order = sample(runners, args.repetitions, args.inner, batch, 123)
            graphs, graph_runners, graph_unsupported = {}, {}, {}
            for name, fn in runners.items():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                try:
                    with torch.cuda.stream(side):
                        for _ in range(3): fn()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=side): out = fn()
                        graph.replay()
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out.float(), reference, atol=3e-3, rtol=3e-3)
                except (RuntimeError, NotImplementedError) as error:
                    graph_unsupported[name] = str(error)
                else:
                    graphs[name] = (graph, out)
                    graph_runners[name] = graph.replay
            replay, graph_order = sample(graph_runners, args.repetitions, args.inner, batch, 124)
            profiles, profile_errors = {}, {}
            for name in ['custom_s256_serial', 'custom_s256_shared', 'flashinfer_cuda_core']:
                try:
                    profiles[name] = trace_runner(runners[name], f'b{batch}_{name}', args.output)
                except Exception as error:
                    profile_errors[name] = {'type': type(error).__name__, 'error': str(error)}
            result['batches'][str(batch)] = {'correctness': correctness,
                'workspace_bytes': {n: p.workspace_bytes for n,p in plans.items()},
                'live_lengths': lengths.tolist(), 'flashinfer_version': fi.version,
                'eager': eager, 'eager_order': order, 'cuda_graph': replay,
                'graph_order': graph_order, 'graph_unsupported': graph_unsupported,
                'profiles': profiles, 'profile_errors': profile_errors}
            (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print('COMPLETED_BATCH', batch, flush=True)
    artifacts = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in args.output.iterdir() if f.is_file()}
    (args.output/'artifact-sha256.json').write_text(json.dumps(artifacts, indent=2)+'\n')


if __name__ == '__main__':
    main()
