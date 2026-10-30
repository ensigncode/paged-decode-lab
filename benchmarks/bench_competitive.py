#!/usr/bin/env python3
"""Compare the two merge implementations with explicit SDPA and paged FlashInfer.

Eager and CUDA-graph replay timings are separate experiments. Unsupported forced
backends are reported with their errors, never assigned timings or silently used
as fallback. Compilation, conversion, planning, profiling and capture are untimed.
"""
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import warnings
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from paged_decode.competitors import FlashInferPlan,identify_sdpa_backend
from paged_decode.metrics import summarize
from paged_decode.reference import gather_dense,expand_gqa,eager_attention,sdpa_attention
from paged_decode.runtime import DecodePlan
from paged_decode.workload import make_metadata,make_tensors


def source_record():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','diff','HEAD','--','src','tests','benchmarks'],text=True)
    names=subprocess.check_output(['git','ls-files','src','tests','benchmarks'],text=True).splitlines()
    return {'revision':sha,'tracked_code_dirty':bool(dirty),
            'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names}}


def measure(fn,inner):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall=time.perf_counter()
    start.record()
    for _ in range(inner): fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)/inner,(time.perf_counter()-wall)*1000/inner


def sample(runners,repetitions,inner,batch,seed):
    raw={name:{'device_ms':[],'wall_ms':[]} for name in runners}
    orders=[];rng=random.Random(seed)
    for _ in range(repetitions):
        order=list(runners);rng.shuffle(order);orders.append(order)
        for name in order:
            device_ms,wall_ms=measure(runners[name],inner)
            raw[name]['device_ms'].append(device_ms);raw[name]['wall_ms'].append(wall_ms)
    return {name:{'summary':summarize(s['device_ms'],s['wall_ms'],batch),'raw':s}
            for name,s in raw.items()},orders


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--context',type=int,default=4096)
    p.add_argument('--heads',type=int,default=32)
    p.add_argument('--kv-heads',type=int,default=8)
    p.add_argument('--dim',type=int,choices=[64,128],default=128)
    p.add_argument('--page-size',type=int,choices=[16,32],default=16)
    p.add_argument('--split-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--repetitions',type=int,default=30)
    p.add_argument('--inner',type=int,default=20)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.repetitions,a.inner)<=0: p.error('positive repetition counts required')
    if not torch.cuda.is_available(): raise SystemExit('A real NVIDIA GPU is required')
    if a.output.exists(): raise SystemExit('Refusing to overwrite an existing report')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    layout,table,lengths=make_metadata(a.batch,a.heads,a.kv_heads,a.dim,a.context,a.page_size,seed=a.seed)
    q,k,v=make_tensors(layout,seed=a.seed)
    serial=DecodePlan(layout,table,lengths,split_tokens=a.split_tokens,merge='serial')
    shared=DecodePlan(layout,table,lengths,split_tokens=a.split_tokens,merge='shared')
    dk,dv,mask=gather_dense(k,v,table,lengths)
    ek,ev=expand_gqa(q,dk,dv)
    del dk,dv
    runners={'cuda_serial_merge':lambda:serial.run(q,k,v),
             'cuda_shared_merge':lambda:shared.run(q,k,v),
             'sdpa_auto_dense':lambda:sdpa_attention(q,ek,ev,mask)}
    unsupported={};external={}
    forced={'math':SDPBackend.MATH,'flash':SDPBackend.FLASH_ATTENTION,
            'efficient':SDPBackend.EFFICIENT_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION}
    def forced_runner(backend):
        def run():
            with sdpa_kernel(backend): return sdpa_attention(q,ek,ev,mask)
        return run
    with torch.inference_mode():
        reference=eager_attention(q,ek,ev,mask)
        dispatch=identify_sdpa_backend(runners['sdpa_auto_dense'])
        for name,backend in forced.items():
            fn=forced_runner(backend)
            try:
                with warnings.catch_warnings(record=True) as notes:
                    warnings.simplefilter('always');output=fn()
                torch.cuda.synchronize()
            except (RuntimeError,NotImplementedError) as error:
                unsupported['sdpa_'+name]={'error':str(error),'type':type(error).__name__}
            else:
                runners['sdpa_'+name+'_dense']=fn
        # Fail if the primary paged competitor cannot run. Tensor-core decode
        # is an additional comparison and may be unsupported on a given GPU.
        for tc in (False,True):
            name='flashinfer_tensor_core' if tc else 'flashinfer_cuda_core'
            try:
                plan=FlashInferPlan(layout,table,lengths,device=q.device,tensor_cores=tc)
                output=plan.run(q,k,v);torch.cuda.synchronize()
            except Exception as error:
                if not tc: raise
                unsupported[name]={'error':str(error),'type':type(error).__name__}
            else:
                runners[name]=lambda plan=plan:plan.run(q,k,v)
                external[name]={'version':plan.version,'backend':plan.backend,
                    'use_tensor_cores':tc,'workspace_bytes':plan.workspace.numel(),
                    'layout':'NHD; same paged k/v tensors; no KV conversion',
                    'output_dtype':str(output.dtype)}
        correctness={}
        for name,fn in runners.items():
            output=fn().float()
            torch.testing.assert_close(output,reference,atol=3e-3,rtol=3e-3)
            correctness[name]={'max_abs_vs_fp32_eager':(output-reference).abs().max().item(),
                               'atol':3e-3,'rtol':3e-3}
            for _ in range(10): fn()
        torch.cuda.synchronize()
        eager,orders=sample(runners,a.repetitions,a.inner,a.batch,a.seed)
        graph_runners={};graphs={};graph_unsupported={}
        for name,fn in runners.items():
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(side):
                    for _ in range(3): fn()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=side): output=fn()
                    graph.replay()
                torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
                torch.testing.assert_close(output.float(),reference,atol=3e-3,rtol=3e-3)
            except (RuntimeError,NotImplementedError) as error:
                graph_unsupported[name]={'error':str(error),'type':type(error).__name__}
            else:
                graphs[name]=(graph,output)
                graph_runners[name]=graph.replay
        graph_results,graph_order=sample(graph_runners,a.repetitions,a.inner,a.batch,a.seed+1)
    record={'schema_version':2,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'source':source_record(),'environment':{'python':platform.python_version(),
            'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'driver':subprocess.check_output(['nvidia-smi'],text=True)},
        'sdpa_dispatch':dispatch,'unsupported_backends':unsupported,
        'external_baselines':external,'correctness':correctness,
        'merge_implementation':shared.merge_implementation,
        'workspace_bytes':{'serial':serial.workspace_bytes,'shared':shared.workspace_bytes},
        'timing_scope':'One static attention decode operator; plans, metadata conversion, compilation, profiling and graph capture excluded.',
        'output_dtypes':'Custom FP32; FlashInfer and SDPA native FP16.',
        'eager':{'results':eager,'trial_order':orders},
        'cuda_graph':{'results':graph_results,'trial_order':graph_order,'unsupported':graph_unsupported},
        'caveats':['SDPA uses pre-gathered, head-expanded dense KV.',
                   'FlashInfer consumes the same paged KV tensors.',
                   'p95 is a percentile of repetition means, not request latency.',
                   'Graph replay isolates launch overhead; it is not a model server benchmark.']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print('SDPA actual dispatch:',dispatch)
    for mode,data in [('eager',eager),('cuda_graph',graph_results)]:
        for name,entry in data.items(): print(mode,name,entry['summary']['device_ms_p50'])
    print('SAVED',a.output)


if __name__=='__main__': main()
