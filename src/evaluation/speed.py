from statistics import mean

from evaluation.base import Evaluator
from metrics import compute_timing, ScoringConfig, request_score, ers


def _pct(xs: list[float], p: float):
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


class SpeedEvaluator(Evaluator):
    def __init__(self, turns, replayer, scoring_cfg: ScoringConfig):
        self.turns = turns
        self.replayer = replayer
        self.scoring_cfg = scoring_cfg

    async def evaluate(self) -> dict:
        items = await self.replayer.run(self.turns)
        per_request = []
        scores = []
        ttfts = []
        tpots = []
        for item in items:
            t = compute_timing(item.result.token_times_ms, item.result.success)
            warm = item.turn.in_warmup
            score = request_score(t.ttft_ms, t.tpot_ms, t.success, self.scoring_cfg)
            if not warm:
                scores.append(score)
                if t.success:
                    if t.ttft_ms is not None:
                        ttfts.append(t.ttft_ms)
                    if t.tpot_ms is not None:
                        tpots.append(t.tpot_ms)
            per_request.append({
                "request_id": item.turn.request_id,
                "conv_id": item.turn.conv_id,
                "turn_idx": item.turn.turn_idx,
                "in_warmup": warm,
                "ttft_ms": t.ttft_ms,
                "tpot_ms": t.tpot_ms,
                "score": None if warm else score,
                "success": t.success,
                "error": item.result.error,
            })
        return {
            "ers": ers(scores),
            "n": len(scores),
            "n_total": len(items),
            "n_success": sum(1 for p in per_request
                             if not p["in_warmup"] and p["success"]),
            "per_request": per_request,
            "ttft_mean_ms": mean(ttfts) if ttfts else None,
            "ttft_p50_ms": _pct(ttfts, 50),
            "ttft_p95_ms": _pct(ttfts, 95),
            "tpot_mean_ms": mean(tpots) if tpots else None,
            "tpot_p50_ms": _pct(tpots, 50),
            "tpot_p95_ms": _pct(tpots, 95),
        }
