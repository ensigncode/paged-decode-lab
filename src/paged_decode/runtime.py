"""Validated, allocation-free replay API around the internal CUDA extension."""
from functools import lru_cache
from pathlib import Path
import math
import torch
from .oracle import Layout, validate_metadata


@lru_cache(maxsize=1)
def load_extension():
    # Explicit preparation keeps compilation out of benchmarks and graph capture.
    from torch.utils.cpp_extension import load
    source = Path(__file__).parent / "csrc"
    return load(
        name="ensigncode_paged_decode_v2",
        sources=[str(source / "bindings.cpp"), str(source / "decode.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=False,
    )


class DecodePlan:
    """Static metadata + reusable workspace for one decode shape.

    table/lengths are CPU integer arrays and are copied during construction.
    KV contents and q can change between runs; metadata requires a new plan.
    The output is borrowed storage, overwritten by the next run. A plan may
    be used on one stream at a time. Use distinct plans for concurrent streams.
    Device metadata and workspace attributes are internal and must not be edited.
    """
    def __init__(self, layout: Layout, table, lengths, *, device="cuda", split_tokens=256, merge="serial"):
        if merge not in ("serial", "shared"):
            raise ValueError("merge must be 'serial' or 'shared'")
        if layout.head_dim not in (64, 128):
            raise ValueError("head_dim must be 64 or 128")
        if layout.page_size not in (16, 32):
            raise ValueError("page_size must be 16 or 32")
        if layout.batch > 65535 or layout.query_heads > 65535:
            raise ValueError("batch/head count exceeds CUDA grid limits")
        if isinstance(split_tokens, bool) or not isinstance(split_tokens, int) or not 1 <= split_tokens <= 1048576:
            raise ValueError("split_tokens must be an integer in [1,1048576]")
        table, lengths = validate_metadata(layout, table, lengths)
        if table.shape[1] > 65536:
            raise ValueError("page table width must not exceed 65536")
        self.max_context = int(lengths.max(initial=0))
        if self.max_context > 1048576:
            raise ValueError("this implementation supports at most 1M context tokens")
        self.splits = max(1, math.ceil(self.max_context / split_tokens))
        if self.splits > 65535:
            raise ValueError("too many context splits")
        self.layout, self.split_tokens = layout, split_tokens
        self.merge = merge
        self.merge_implementation = ("shared" if self.splits * 4 <= 4096 else "serial_fallback") if merge == "shared" else "serial"
        self._merge_mode = int(self.merge_implementation == "shared")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DecodePlan requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._table_cpu, self._lengths_cpu = table, lengths
        self._table = torch.tensor(table, device=self.device)
        self._lengths = torch.tensor(lengths, device=self.device)
        shape = (layout.batch, layout.query_heads, self.splits * 4)
        self._numerator = torch.empty((*shape, layout.head_dim), device=self.device, dtype=torch.float32)
        self._stats = torch.empty((*shape, 2), device=self.device, dtype=torch.float32)
        self._out = torch.empty((layout.batch, layout.query_heads, layout.head_dim),
                                dtype=torch.float32, device=self.device)
        self._extension = load_extension()

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self._numerator, self._stats))

    def run(self, q, k, v):
        layout = self.layout
        expected_q = (layout.batch, layout.query_heads, layout.head_dim)
        expected_k = (layout.physical_pages, layout.page_size, layout.kv_heads, layout.head_dim)
        if tuple(q.shape) != expected_q or tuple(k.shape) != expected_k or tuple(v.shape) != expected_k:
            raise ValueError("input shape does not match plan")
        if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError("q/k/v must share float16 or float32 dtype")
        for tensor in (q, k, v):
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError("inputs must be contiguous on the plan device")
            if tensor.requires_grad:
                raise ValueError("inference only: detach tensors requiring gradients")
        self._extension.decode(q, k, v, self._table, self._lengths,
                               self._numerator, self._stats, self._out, self.split_tokens, self._merge_mode)
        return self._out
