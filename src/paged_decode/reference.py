"""PyTorch baselines. Neither baseline is a paged production serving engine."""
import math
import torch
import torch.nn.functional as F


def gather_dense(k, v, table, lengths):
    """Padding is filled with zeros; invalid page ids are never dereferenced."""
    batch, max_context = len(lengths), max(1, int(max(lengths)))
    shape = (batch, k.shape[2], max_context, k.shape[3])
    dense_k, dense_v = k.new_zeros(shape), v.new_zeros(shape)
    mask = torch.zeros((batch, 1, 1, max_context), dtype=torch.bool, device=k.device)
    for b, length in enumerate(lengths):
        length = int(length)
        if length == 0:
            continue
        logical = torch.arange(length, device=k.device)
        pages = torch.as_tensor(table[b, :math.ceil(length / k.shape[1])].copy(),
                                device=k.device, dtype=torch.long)
        physical = pages[logical // k.shape[1]]
        offsets = logical % k.shape[1]
        dense_k[b, :, :length] = k[physical, offsets].transpose(0, 1)
        dense_v[b, :, :length] = v[physical, offsets].transpose(0, 1)
        mask[b, :, :, :length] = True
    # Explicit repetition makes the baseline portable across SDPA GQA backends;
    # benchmark reports this memory expansion separately from cache gathering.
    return dense_k, dense_v, mask


def expand_gqa(q, dense_k, dense_v):
    repeats = q.shape[1] // dense_k.shape[1]
    return (dense_k.repeat_interleave(repeats, dim=1),
            dense_v.repeat_interleave(repeats, dim=1))


def eager_attention(q, dense_k, dense_v, mask):
    logits = (q.float().unsqueeze(2) @ dense_k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~mask, -torch.inf)
    # Define an all-empty sequence as zero rather than NaN.
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    return (weights @ dense_v.float()).squeeze(2)


def sdpa_attention(q, dense_k, dense_v, mask):
    return F.scaled_dot_product_attention(q.unsqueeze(2), dense_k, dense_v,
                                         attn_mask=mask, dropout_p=0.0).squeeze(2)
