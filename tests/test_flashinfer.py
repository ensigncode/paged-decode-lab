"""Optional external-library acceptance. Requires an installed FlashInfer build."""
import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("flashinfer")
if not torch.cuda.is_available():
    pytest.skip("NVIDIA GPU required",allow_module_level=True)
from paged_decode.competitors import FlashInferPlan
from paged_decode.oracle import dense_attention
from paged_decode.workload import make_metadata,make_tensors
pytestmark=pytest.mark.gpu


@pytest.mark.parametrize("dim,page,heads,kv_heads",[(64,16,8,8),(128,32,32,8),(128,16,8,1)])
def test_external_paged_reference_matches_independent_oracle(dim,page,heads,kv_heads):
    layout,table,lengths=make_metadata(3,heads,kv_heads,dim,129,page,seed=87)
    q,k,v=make_tensors(layout,seed=88)
    plan=FlashInferPlan(layout,table,lengths,device=q.device)
    actual=plan.run(q,k,v).float().cpu().numpy()
    expected=dense_attention(*(t.cpu().numpy() for t in (q,k,v)),table,lengths)
    np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=3e-3)
