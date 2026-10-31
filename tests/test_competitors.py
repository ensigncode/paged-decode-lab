import numpy as np
import pytest

torch = pytest.importorskip("torch")
from paged_decode.competitors import paged_csr
from paged_decode.oracle import Layout


def test_csr_preserves_fragmented_shared_pages_and_partial_lengths():
    layout=Layout(2,8,2,64,16,4)
    offsets,indices,last=paged_csr(layout,[[2,0,1],[2,3,-1]],[33,17])
    np.testing.assert_array_equal(offsets,[0,3,5])
    np.testing.assert_array_equal(indices,[2,0,1,2,3])
    np.testing.assert_array_equal(last,[1,1])
    assert offsets.dtype == indices.dtype == last.dtype == np.int32


def test_csr_rejects_empty_external_comparison_rows():
    with pytest.raises(ValueError,match="nonempty"):
        paged_csr(Layout(1,4,1,64,16,1),[[0]],[0])
