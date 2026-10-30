import pytest
from paged_decode.metrics import percentile, summarize


def test_percentiles_are_interpolated():
    assert percentile([1,2,3,4,5], .95) == pytest.approx(4.8)
    assert percentile([5], .95) == 5


def test_throughput_uses_sequences_per_decode_step():
    summary = summarize([1.,2.,3.], [2.,3.,4.], batch=8)
    assert summary["device_ms_p50"] == 2
    assert summary["sequence_tokens_per_second"] == 4000
    assert summary["wall_ms_p50"] == 3


@pytest.mark.parametrize("device,wall", [([],[]), ([0.],[1.]), ([1.],[])])
def test_invalid_samples_are_rejected(device,wall):
    with pytest.raises(ValueError):
        summarize(device,wall,1)
