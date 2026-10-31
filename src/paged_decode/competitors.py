"""Adapters for measured external baselines; no external kernel code is vendored."""
import math
import numpy as np
import torch
from .oracle import validate_metadata


def paged_csr(layout, table, lengths):
    table,lengths = validate_metadata(layout,table,lengths)
    if np.any(lengths == 0):
        raise ValueError("FlashInfer comparison requires nonempty contexts")
    counts = (lengths.astype(np.int64) + layout.page_size - 1) // layout.page_size
    offsets = np.concatenate(([0],np.cumsum(counts)))
    if int(offsets[-1]) > np.iinfo(np.int32).max:
        raise ValueError("CSR page list exceeds int32 capacity")
    indices = np.concatenate([table[b,:int(count)] for b,count in enumerate(counts)])
    last = (lengths.astype(np.int64)-1) % layout.page_size + 1
    return offsets.astype(np.int32),indices.astype(np.int32),last.astype(np.int32)


class FlashInferPlan:
    """Use the same NHD k/v tensors, without gathering or copying KV contents.

    Plan construction, metadata upload and the 128 MiB external workspace are
    excluded from steady-state timings and reported separately. The external
    baseline uses its native FP16 output; our CUDA operator emits FP32.
    """
    def __init__(self,layout,table,lengths,*,device, tensor_cores=False):
        import flashinfer
        offsets,indices,last = paged_csr(layout,table,lengths)
        self.version = flashinfer.__version__
        self.tensor_cores = tensor_cores
        self.backend = "fa2"
        self.workspace = torch.zeros(128*1024*1024,dtype=torch.uint8,device=device)
        self.offsets = torch.tensor(offsets,device=device)
        self.indices = torch.tensor(indices,device=device)
        self.last = torch.tensor(last,device=device)
        self.out = torch.empty((layout.batch,layout.query_heads,layout.head_dim),
                               device=device,dtype=torch.float16)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,"NHD",use_cuda_graph=True,use_tensor_cores=tensor_cores,
            paged_kv_indptr_buffer=self.offsets,
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last,backend=self.backend)
        self.wrapper.plan(self.offsets,self.indices,self.last,layout.query_heads,
            layout.kv_heads,layout.head_dim,layout.page_size,
            q_data_type=torch.float16,kv_data_type=torch.float16,
            pos_encoding_mode="NONE",sm_scale=1/math.sqrt(layout.head_dim))

    def run(self,q,k,v):
        if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError("FlashInfer comparison adapter requires FP16 inputs")
        return self.wrapper.run(q,(k,v),out=self.out)


def identify_sdpa_backend(fn):
    """Record actual framework dispatch names, outside all timing regions."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        fn()
        torch.cuda.synchronize()
    operators = sorted({event.key for event in trace.key_averages()
                        if "attention" in event.key.lower()})
    families = {"math":"_attention_math", "flash":"_flash_attention",
                "efficient":"_efficient_attention", "cudnn":"_cudnn_attention"}
    matches = [name for name,token in families.items() if any(token in op for op in operators)]
    return {"identified_backend": matches[0] if len(matches)==1 else "unresolved",
            "observed_operators":operators,
            "method":"CPU profiler records actual SDPA dispatch; not a GPU bandwidth profile"}
