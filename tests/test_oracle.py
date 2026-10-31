import numpy as np
import pytest
from paged_decode.oracle import (Layout, validate_metadata, dense_attention,
                                streaming_attention, partial_softmax, merge_partials)
from paged_decode.workload import make_metadata


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("split", [1, 17, 256])
@pytest.mark.parametrize("magnitude", [1.0, 100.0])
def test_partitioned_attention_matches_dense(dim, split, magnitude):
    layout, table, lengths = make_metadata(3, 4, 2, dim, 35, 16)
    lengths[:] = [0, 1, 35]
    # Restore all entries because make_metadata may have padded a ragged row.
    table = np.random.default_rng(9).permutation(layout.physical_pages).reshape(3, -1)
    rng = np.random.default_rng(4)
    q = rng.normal(size=(3, 4, dim)) * magnitude
    k = rng.normal(size=(layout.physical_pages, 16, 2, dim))
    v = rng.normal(size=k.shape)
    expected = dense_attention(q, k, v, table, lengths)
    actual = streaming_attention(q, k, v, table, lengths, split)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("bad_lengths", [[-1, 1], [33, 1]])
def test_invalid_lengths_rejected(bad_lengths):
    layout = Layout(2, 4, 2, 64, 16, 4)
    with pytest.raises(ValueError):
        validate_metadata(layout, np.array([[0, 1], [2, 3]]), np.array(bad_lengths))


def test_padding_is_not_a_live_page():
    layout = Layout(1, 4, 2, 64, 16, 2)
    table, lengths = validate_metadata(layout, np.array([[0, -1]]), np.array([1]))
    assert table.dtype == lengths.dtype == np.int32
    with pytest.raises(ValueError):
        validate_metadata(layout, np.array([[0, -1]]), np.array([17]))
    with pytest.raises(ValueError):
        validate_metadata(layout, np.array([[2, 0]]), np.array([1]))
    with pytest.raises(TypeError):
        validate_metadata(layout, np.array([[0., 1.]]), np.array([1]))


def test_shared_prefix_pages():
    rng = np.random.default_rng(8)
    q = rng.normal(size=(2, 4, 64))
    k, v = rng.normal(size=(2, 3, 16, 2, 64))
    table = np.array([[2, 0], [2, 1]])
    lengths = np.array([17, 31])
    np.testing.assert_allclose(streaming_attention(q,k,v,table,lengths,7),
                               dense_attention(q,k,v,table,lengths), atol=1e-12)


def test_softmax_merge_survives_extreme_logits_and_empty_partitions():
    scores = np.array([-10000., 10000., 9999.])
    values = np.arange(12, dtype=float).reshape(3, 4)
    parts = [partial_softmax(scores[:0], values[:0]),
             partial_softmax(scores[:1], values[:1]),
             partial_softmax(scores[1:], values[1:])]
    weights = np.exp(scores - scores.max())
    np.testing.assert_allclose(merge_partials(parts), weights @ values / weights.sum())
    np.testing.assert_array_equal(merge_partials(parts[:1]), np.zeros(4))


def test_metadata_validation_copies_caller_arrays():
    layout = Layout(1, 4, 2, 64, 16, 2)
    table, lengths = np.array([[0, 1]]), np.array([17])
    copied_table, copied_lengths = validate_metadata(layout, table, lengths)
    table[:] = -1
    lengths[:] = 0
    np.testing.assert_array_equal(copied_table, [[0, 1]])
    np.testing.assert_array_equal(copied_lengths, [17])


@pytest.mark.parametrize("value", [True, False, 1.5, "4", np.float64(4)])
def test_layout_requires_integer_dimensions(value):
    with pytest.raises(TypeError):
        Layout(value,4,2,64,16,2)


@pytest.mark.parametrize("value", [0,-1])
def test_layout_requires_positive_dimensions(value):
    with pytest.raises(ValueError):
        Layout(value,4,2,64,16,2)


def test_invalid_head_ratio_is_rejected():
    with pytest.raises(ValueError):
        Layout(1,3,2,64,16,2)
