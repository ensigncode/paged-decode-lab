"""Report helpers kept independent of CUDA for offline verification."""
import math
import statistics


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError("nonempty values and quantile in [0,1] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(device_ms, wall_ms, batch):
    if len(device_ms) != len(wall_ms) or not device_ms or min(device_ms + wall_ms) <= 0:
        raise ValueError("paired positive timing samples required")
    median = statistics.median(device_ms)
    return {
        "sample_count": len(device_ms),
        "device_ms_p50": median,
        "device_ms_p95": percentile(device_ms, 0.95),
        "device_ms_min": min(device_ms),
        "wall_ms_p50": statistics.median(wall_ms),
        "decode_steps_per_second": 1000.0 / median,
        "sequence_tokens_per_second": batch * 1000.0 / median,
    }
