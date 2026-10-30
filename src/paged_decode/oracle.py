"""Small NumPy oracle; intentionally independent of the CUDA implementation.

All calculations use float64 so the oracle can expose FP32 accumulation errors.
The CPU streaming implementation also exercises the split-softmax merge algebra.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class Layout:
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    physical_pages: int

    def __post_init__(self):
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in vars(self).values()):
            raise TypeError("layout dimensions must be integers, not booleans")
        if min(vars(self).values()) <= 0:
            raise ValueError("layout dimensions must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")


def validate_metadata(layout, table, lengths):
    table, lengths = np.asarray(table), np.asarray(lengths)
    if table.ndim != 2 or table.shape[0] != layout.batch:
        raise ValueError("page table must have shape [batch, logical_pages]")
    if lengths.shape != (layout.batch,):
        raise ValueError("lengths must have shape [batch]")
    if table.dtype.kind not in "iu" or lengths.dtype.kind not in "iu":
        raise TypeError("metadata must contain integers")
    if np.any(lengths < 0) or np.any(lengths > table.shape[1] * layout.page_size):
        raise ValueError("sequence length exceeds page table capacity")
    if max(table.shape[1], layout.physical_pages) > np.iinfo(np.int32).max:
        raise ValueError("metadata exceeds int32 addressing limits")
    for b, length in enumerate(lengths):
        count = (int(length) + layout.page_size - 1) // layout.page_size
        used = table[b, :count]
        if np.any(used < 0) or np.any(used >= layout.physical_pages):
            raise ValueError("live logical page maps outside physical cache")
    return table.astype(np.int32, copy=True), lengths.astype(np.int32, copy=True)


def gather_cache(cache, pages, length, kv_head):
    """Gather live tokens only; padding page ids may be -1."""
    if length == 0:
        return np.empty((0, cache.shape[-1]), dtype=np.float64)
    positions = np.arange(length)
    return cache[pages[positions // cache.shape[1]],
                 positions % cache.shape[1], kv_head].astype(np.float64)


def dense_attention(q, k, v, table, lengths):
    output = np.zeros_like(q, dtype=np.float64)
    ratio = q.shape[1] // k.shape[2]
    for b, length in enumerate(lengths):
        for h in range(q.shape[1]):
            if not length:
                continue
            keys = gather_cache(k, table[b], int(length), h // ratio)
            values = gather_cache(v, table[b], int(length), h // ratio)
            scores = keys @ q[b, h].astype(np.float64) / np.sqrt(q.shape[-1])
            weights = np.exp(scores - scores.max())
            output[b, h] = (weights @ values) / weights.sum()
    return output


def partial_softmax(scores, values):
    """Return (maximum, denominator, unnormalized weighted value sum)."""
    m, denominator = -np.inf, 0.0
    numerator = np.zeros(values.shape[-1], dtype=np.float64)
    for score, value in zip(scores, values):
        new_m = max(m, score)
        alpha = np.exp(m - new_m)
        beta = np.exp(score - new_m)
        numerator = numerator * alpha + beta * value
        denominator = denominator * alpha + beta
        m = new_m
    return m, denominator, numerator


def merge_partials(partials):
    """The zero-length partition is the identity, including all-empty inputs."""
    active = [part for part in partials if part[1] > 0]
    if not active:
        return np.zeros_like(partials[0][2])
    maximum = max(part[0] for part in active)
    denominator = sum(np.exp(m - maximum) * l for m, l, _ in active)
    numerator = sum(np.exp(m - maximum) * n for m, _, n in active)
    return numerator / denominator


def streaming_attention(q, k, v, table, lengths, split_tokens=256):
    if split_tokens <= 0:
        raise ValueError("split_tokens must be positive")
    output = np.zeros_like(q, dtype=np.float64)
    ratio = q.shape[1] // k.shape[2]
    for b, length in enumerate(lengths):
        for h in range(q.shape[1]):
            keys = gather_cache(k, table[b], int(length), h // ratio)
            values = gather_cache(v, table[b], int(length), h // ratio)
            scores = keys @ q[b, h].astype(np.float64) / np.sqrt(q.shape[-1])
            partials = []
            for start in range(0, max(int(length), 1), split_tokens):
                stop = min(start + split_tokens, int(length))
                # Four warps own disjoint strided token sets in each CTA.
                for warp in range(4):
                    partials.append(partial_softmax(scores[start + warp:stop:4],
                                                   values[start + warp:stop:4]))
            output[b, h] = merge_partials(partials)
    return output
