"""Deterministic fragmented page maps, without downloading model weights."""
import math
import numpy as np
from .oracle import Layout


def make_metadata(batch, heads, kv_heads, dim, context, page_size, *, seed=123, ragged=True):
    if min(batch, heads, kv_heads, dim, context, page_size) <= 0:
        raise ValueError("workload dimensions must be positive")
    rng = np.random.default_rng(seed)
    width = math.ceil(context / page_size)
    # No physical page sharing in this workload. Production caches may share
    # prefix pages; the runtime supports that but has no copy-on-write allocator.
    table = rng.permutation(batch * width).reshape(batch, width).astype(np.int32)
    lengths = rng.integers(max(1, context // 4), context + 1, size=batch, dtype=np.int32)
    if not ragged:
        lengths.fill(context)
    lengths[0] = context
    layout = Layout(batch, heads, kv_heads, dim, page_size, batch * width)
    for b, length in enumerate(lengths):
        table[b, math.ceil(int(length) / page_size):] = -1
    return layout, table, lengths


def make_tensors(layout, *, device="cuda", dtype=None, seed=123):
    import torch
    dtype = torch.float16 if dtype is None else dtype
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((layout.batch, layout.query_heads, layout.head_dim),
                    device=device, dtype=dtype, generator=generator)
    shape = (layout.physical_pages, layout.page_size, layout.kv_heads, layout.head_dim)
    k = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return q, k, v
