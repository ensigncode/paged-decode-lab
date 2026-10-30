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


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("split", [1, 33, 256])
def test_ragged_decode(dim, dtype, page_size, split):
    layout = Layout(4, 8, 2, dim, page_size, 32)
    table = np.random.default_rng(5).permutation(32).reshape(4, 8).astype(np.int32)
    lengths = np.array([0, 1, page_size + 1, 127], dtype=np.int32)
    for b, length in enumerate(lengths):
        table[b, (int(length) + page_size - 1) // page_size:] = -1
    q, k, v = make_tensors(layout, dtype=dtype)
    plan = DecodePlan(layout, table, lengths, split_tokens=split)
    actual = plan.run(q, k, v).cpu().numpy()
    expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)), table, lengths)
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
    assert np.isfinite(actual).all()


def test_nondefault_stream_and_graph_capture():
    layout = Layout(1, 8, 2, 64, 16, 8)
    table, lengths = np.arange(8, dtype=np.int32)[None], np.array([127], dtype=np.int32)
    q, k, v = make_tensors(layout)
    plan = DecodePlan(layout, table, lengths)
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


def test_large_logits_and_repeated_pages():
    layout = Layout(2, 8, 1, 128, 16, 4)
    table = np.array([[2, 0, 1, 3], [2, 0, 1, 3]], dtype=np.int32)
    lengths = np.array([64, 37], dtype=np.int32)
    q, k, v = make_tensors(layout, dtype=torch.float32)
    q *= 100
    plan = DecodePlan(layout, table, lengths, split_tokens=17)
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


def test_all_empty_contexts_and_long_partition_tail():
    for lengths in (np.array([0,0],dtype=np.int32), np.array([8193,17],dtype=np.int32)):
        layout = Layout(2,4,4,64,16,1026)
        table = np.arange(1026,dtype=np.int32).reshape(2,513)
        q,k,v = make_tensors(layout, dtype=torch.float32)
        plan = DecodePlan(layout,table,lengths,split_tokens=257)
        actual = plan.run(q,k,v).cpu().numpy()
        expected = dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,lengths)
        np.testing.assert_allclose(actual,expected,atol=5e-4,rtol=5e-4)
