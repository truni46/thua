from dataclasses import dataclass


@dataclass
class RequestTiming:
    ttft_ms: float | None
    tpot_ms: float | None
    n_tokens: int
    success: bool


def compute_timing(token_times_ms: list[float], success: bool) -> RequestTiming:
    if not success or not token_times_ms:
        return RequestTiming(None, None, 0, False)
    ttft = token_times_ms[0]
    n = len(token_times_ms)
    tpot = (token_times_ms[-1] - token_times_ms[0]) / (n - 1) if n > 1 else None
    return RequestTiming(ttft, tpot, n, True)


@dataclass(frozen=True)
class ScoringConfig:
    f_ttft: float = 10.0
    c_ttft: float = 400.0
    f_tpot: float = 1.0
    c_tpot: float = 10.0
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


KEY_METRICS = [
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
]


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        if not name_part:
            continue
        name = name_part.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out
