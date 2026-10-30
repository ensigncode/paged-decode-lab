"""CPU integration checks for the actual PyTorch benchmark references."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from paged_decode.oracle import Layout, dense_attention
from paged_decode.reference import gather_dense, expand_gqa, eager_attention, sdpa_attention
from paged_decode.workload import make_tensors


@pytest.mark.parametrize("kv_heads", [1,2,4])
@pytest.mark.parametrize("lengths", [[0,0,0], [0,1,35]])
def test_torch_references_against_independent_numpy(kv_heads, lengths):
    layout = Layout(3,4,kv_heads,64,16,9)
    table = np.random.default_rng(6).permutation(9).reshape(3,3).astype(np.int32)
    lengths = np.array(lengths, dtype=np.int32)
    for b, length in enumerate(lengths):
        table[b,(int(length)+15)//16:] = -1
    q,k,v = make_tensors(layout, device="cpu", dtype=torch.float32)
    dk,dv,mask = gather_dense(k,v,table,lengths)
    ek,ev = expand_gqa(q,dk,dv)
    oracle = dense_attention(q.numpy(),k.numpy(),v.numpy(),table,lengths)
    for output in (eager_attention(q,ek,ev,mask), sdpa_attention(q,ek,ev,mask)):
        np.testing.assert_allclose(output.numpy(),oracle,atol=2e-6,rtol=2e-5)
        assert torch.isfinite(output).all()
