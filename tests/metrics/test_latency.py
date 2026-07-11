import math
from metrics.latency import compute_timing, RequestTiming


def test_ttft_is_first_token_time():
    t = compute_timing([120.0, 145.0, 170.0], True)
    assert t.ttft_ms == 120.0


def test_tpot_is_mean_interval():
    # (170-120)/(3-1) = 25
    t = compute_timing([120.0, 145.0, 170.0], True)
    assert math.isclose(t.tpot_ms, 25.0, rel_tol=1e-9)
    assert t.n_tokens == 3


def test_single_token_has_no_tpot():
    t = compute_timing([120.0], True)
    assert t.ttft_ms == 120.0
    assert t.tpot_ms is None


def test_failure_is_all_none():
    t = compute_timing([120.0, 130.0], False)
    assert t == RequestTiming(None, None, 0, False)


def test_empty_tokens_is_failure_shape():
    t = compute_timing([], True)
    assert t.success is False
    assert t.ttft_ms is None
