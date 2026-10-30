"""GPU acceptance tests. A skipped module does not validate the CUDA kernels."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from paged_decode.runtime import DecodePlan
from paged_decode.oracle import Layout, dense_attention
from paged_decode.workload import make_tensors

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("merge", ["serial", "shared"])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("split", [1, 33, 256])
def test_ragged_decode(dim, dtype, page_size, split, merge):
    layout = Layout(4, 8, 2, dim, page_size, 32)
    table = np.random.default_rng(5).permutation(32).reshape(4, 8).astype(np.int32)
    lengths = np.array([0, 1, page_size + 1, 127], dtype=np.int32)
    for b, length in enumerate(lengths):
        table[b, (int(length) + page_size - 1) // page_size:] = -1
    q, k, v = make_tensors(layout, dtype=dtype)
    plan = DecodePlan(layout, table, lengths, split_tokens=split, merge=merge)
    actual = plan.run(q, k, v).cpu().numpy()
    expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)), table, lengths)
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("merge", ["serial", "shared"])
def test_nondefault_stream_and_graph_capture(merge):
    layout = Layout(1, 8, 2, 64, 16, 8)
    table, lengths = np.arange(8, dtype=np.int32)[None], np.array([127], dtype=np.int32)
    q, k, v = make_tensors(layout)
    plan = DecodePlan(layout, table, lengths, merge=merge)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = plan.run(q, k, v).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = plan.run(q, k, v)
        graph.replay()
        torch.testing.assert_close(output, expected)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@pytest.mark.parametrize("merge", ["serial", "shared"])
def test_large_logits_and_repeated_pages(merge):
    layout = Layout(2, 8, 1, 128, 16, 4)
    table = np.array([[2, 0, 1, 3], [2, 0, 1, 3]], dtype=np.int32)
    lengths = np.array([64, 37], dtype=np.int32)
    q, k, v = make_tensors(layout, dtype=torch.float32)
    q *= 100
    plan = DecodePlan(layout, table, lengths, split_tokens=17, merge=merge)
    actual = plan.run(q,k,v).cpu().numpy()
    expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)), table, lengths)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-3)


def test_rejects_invalid_inputs():
    layout = Layout(1, 8, 2, 64, 16, 8)
    plan = DecodePlan(layout, np.arange(8, dtype=np.int32)[None], np.array([127]))
    q, k, v = make_tensors(layout)
    with pytest.raises(TypeError):
        plan.run(q.double(), k, v)
    with pytest.raises(ValueError):
        plan.run(q.transpose(1,2), k, v)
    with pytest.raises(ValueError):
        plan.run(q.requires_grad_(), k, v)


@pytest.mark.parametrize("merge", ["serial", "shared"])
def test_all_empty_contexts_and_long_partition_tail(merge):
    for lengths in (np.array([0,0],dtype=np.int32), np.array([8193,17],dtype=np.int32)):
        layout = Layout(2,4,4,64,16,1026)
        table = np.arange(1026,dtype=np.int32).reshape(2,513)
        q,k,v = make_tensors(layout, dtype=torch.float32)
        plan = DecodePlan(layout,table,lengths,split_tokens=257, merge=merge)
        actual = plan.run(q,k,v).cpu().numpy()
        expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,lengths)
        np.testing.assert_allclose(actual,expected,atol=5e-4,rtol=5e-4)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("merge", ["serial", "shared"])
def test_randomized_layout_and_content(seed, merge):
    rng = np.random.default_rng(seed)
    dim, page = int(rng.choice([64,128])), int(rng.choice([16,32]))
    heads = int(rng.choice([1,4,8]))
    kv_heads = int(rng.choice([h for h in [1,2,4,8] if heads % h == 0]))
    layout = Layout(3,heads,kv_heads,dim,page,20)
    lengths = np.array([0, int(rng.integers(1,64)), 257], dtype=np.int32)
    table = rng.integers(0,20,size=(3,(257+page-1)//page),dtype=np.int32)
    dtype = torch.float16 if seed % 2 else torch.float32
    q,k,v = make_tensors(layout,dtype=dtype,seed=seed+200)
    if seed % 3 == 0:
        q.zero_()  # Uniform attention should reduce to a mean over logical tokens.
    plan = DecodePlan(layout,table,lengths,split_tokens=int(rng.choice([1,31,128,511])),merge=merge)
    actual = plan.run(q,k,v).cpu().numpy()
    expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,lengths)
    np.testing.assert_allclose(actual,expected,atol=5e-4,rtol=5e-4)


@pytest.mark.parametrize("merge", ["serial", "shared"])
def test_changing_decode_across_page_boundaries(merge):
    layout = Layout(1,4,2,64,16,4)
    table = np.array([[2,0,3,1]],dtype=np.int32)
    q,k,v = make_tensors(layout,dtype=torch.float32,seed=333)
    # Plans intentionally have immutable metadata; each new token count needs
    # preparation, while the cache and query contents can change on every run.
    for length in [0,1,15,16,17,31,32,33,63,64]:
        q.add_(0.01)
        if length:
            page = int(table[0,(length-1)//16])
            v[page,(length-1)%16].mul_(0.9)
        plan = DecodePlan(layout,table,np.array([length]),split_tokens=13,merge=merge)
        actual = plan.run(q,k,v).cpu().numpy()
        expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,[length])
        np.testing.assert_allclose(actual,expected,atol=5e-4,rtol=5e-4)


def test_shared_merge_resource_bound_uses_explicit_fallback():
    layout = Layout(1,4,1,64,16,70)
    table = np.arange(70,dtype=np.int32)[None]
    lengths = np.array([1100],dtype=np.int32)
    q,k,v = make_tensors(layout,dtype=torch.float32)
    plan = DecodePlan(layout,table,lengths,split_tokens=1,merge="shared")
    assert plan.merge_implementation == "serial_fallback"
    expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,lengths)
    np.testing.assert_allclose(plan.run(q,k,v).cpu().numpy(),expected,atol=5e-4,rtol=5e-4)


def test_rejects_same_shape_noncontiguous_and_wrong_device():
    layout = Layout(1,8,2,64,16,8)
    plan = DecodePlan(layout,np.arange(8,dtype=np.int32)[None],np.array([127]))
    q,k,v = make_tensors(layout)
    strided = torch.empty((1,8,128),device=q.device,dtype=q.dtype)[...,::2]
    assert strided.shape == q.shape and not strided.is_contiguous()
    with pytest.raises(ValueError):
        plan.run(strided,k,v)
    with pytest.raises(ValueError):
        plan.run(q.cpu(),k,v)
