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
