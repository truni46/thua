import math
from metrics.scoring import ScoringConfig, s_ttft, s_tpot, request_score, ers

CFG = ScoringConfig()


def test_s_ttft_at_floor_is_one():
    assert s_ttft(100, CFG) == 1.0


def test_s_ttft_at_ceiling_is_zero():
    assert s_ttft(1500, CFG) == 0.0


def test_s_ttft_midpoint_gamma_squared():
    # (1500-800)/1400 = 0.5 -> 0.25 with gamma=2
    assert math.isclose(s_ttft(800, CFG), 0.25, rel_tol=1e-9)


def test_s_tpot_none_is_zero():
    assert s_tpot(None, CFG) == 0.0


def test_s_tpot_midpoint():
    # (45-32.5)/25 = 0.5 -> 0.25
    assert math.isclose(s_tpot(32.5, CFG), 0.25, rel_tol=1e-9)


def test_request_score_best_case():
    assert math.isclose(request_score(100, 20, True, CFG), 1.0, rel_tol=1e-9)


def test_request_score_failure_is_zero():
    assert request_score(120, 25, False, CFG) == 0.0


def test_request_score_none_ttft_is_zero():
    assert request_score(None, 25, True, CFG) == 0.0


def test_ers_average():
    assert math.isclose(ers([1.0, 0.0, 0.5]), 0.5, rel_tol=1e-9)


def test_ers_empty_is_zero():
    assert ers([]) == 0.0
