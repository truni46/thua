import math
from metrics import delta, penalty, final_score, DEFAULT_BASELINE


def test_default_baseline():
    assert DEFAULT_BASELINE == 0.4


def test_delta():
    assert math.isclose(delta(0.4, 0.32), 0.08, rel_tol=1e-9)


def test_penalty_no_drop():
    assert penalty(0.10) == 1.0
    assert penalty(0.05) == 1.0


def test_penalty_full_cutoff():
    assert penalty(0.16) == 0.0
    assert penalty(0.20) == 0.0


def test_penalty_linear_midpoint():
    # Δ=0.13 -> 1 - (0.03/0.06) = 0.5
    assert math.isclose(penalty(0.13), 0.5, rel_tol=1e-9)


def test_final_score():
    # ERS 0.6, Δ 0.13 -> 100*0.6*0.5 = 30
    assert math.isclose(final_score(0.6, 0.13), 30.0, rel_tol=1e-9)
