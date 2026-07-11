DEFAULT_BASELINE = 0.4


def delta(baseline: float, acc: float) -> float:
    return baseline - acc


def penalty(d: float) -> float:
    if d <= 0.10:
        return 1.0
    if d >= 0.16:
        return 0.0
    return 1.0 - (d - 0.10) / 0.06


def final_score(ers_value: float, d: float) -> float:
    return 100.0 * ers_value * penalty(d)
