import math
from metrics import ScoringConfig, s_ttft, s_tpot, request_score, ers

CFG = ScoringConfig()


def test_s_ttft_at_floor_is_one():
    assert s_ttft(10, CFG) == 1.0


def test_s_ttft_at_ceiling_is_zero():
    assert s_ttft(400, CFG) == 0.0


def test_s_ttft_midpoint_gamma_squared():
    # (400-205)/390 = 0.5 -> 0.25 with gamma=2
    assert math.isclose(s_ttft(205, CFG), 0.25, rel_tol=1e-9)


def test_s_tpot_none_is_zero():
    assert s_tpot(None, CFG) == 0.0


def test_s_tpot_midpoint():
    # (10-5.5)/9 = 0.5 -> 0.25
    assert math.isclose(s_tpot(5.5, CFG), 0.25, rel_tol=1e-9)


def test_request_score_best_case():
    assert math.isclose(request_score(10, 1, True, CFG), 1.0, rel_tol=1e-9)


def test_request_score_failure_is_zero():
    assert request_score(50, 3, False, CFG) == 0.0


def test_request_score_none_ttft_is_zero():
    assert request_score(None, 3, True, CFG) == 0.0


def test_ers_average():
    assert math.isclose(ers([1.0, 0.0, 0.5]), 0.5, rel_tol=1e-9)


def test_ers_empty_is_zero():
    assert ers([]) == 0.0
