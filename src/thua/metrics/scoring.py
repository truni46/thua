from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    f_ttft: float = 100.0
    c_ttft: float = 1500.0
    f_tpot: float = 20.0
    c_tpot: float = 45.0
    gamma: float = 2.0
    w: float = 0.5


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def s_ttft(ttft_ms: float, cfg: ScoringConfig) -> float:
    x = _clamp01((cfg.c_ttft - ttft_ms) / (cfg.c_ttft - cfg.f_ttft))
    return x ** cfg.gamma


def s_tpot(tpot_ms: float | None, cfg: ScoringConfig) -> float:
    if tpot_ms is None:
        return 0.0
    x = _clamp01((cfg.c_tpot - tpot_ms) / (cfg.c_tpot - cfg.f_tpot))
    return x ** cfg.gamma


def request_score(ttft_ms: float | None, tpot_ms: float | None,
                  success: bool, cfg: ScoringConfig) -> float:
    if not success or ttft_ms is None:
        return 0.0
    return cfg.w * s_ttft(ttft_ms, cfg) + (1.0 - cfg.w) * s_tpot(tpot_ms, cfg)


def ers(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0
