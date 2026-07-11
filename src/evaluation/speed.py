from statistics import mean

from evaluation.base import Evaluator
from metrics.latency import compute_timing
from metrics.scoring import ScoringConfig, request_score, ers


class SpeedEvaluator(Evaluator):
    def __init__(self, requests, replayer, scoring_cfg: ScoringConfig):
        self.requests = requests
        self.replayer = replayer
        self.scoring_cfg = scoring_cfg

    async def evaluate(self) -> dict:
        items = await self.replayer.run(self.requests)
        per_request = []
        scores = []
        ttfts = []
        tpots = []
        for item in items:
            t = compute_timing(item.result.token_times_ms, item.result.success)
            score = request_score(t.ttft_ms, t.tpot_ms, t.success, self.scoring_cfg)
            scores.append(score)
            if t.success:
                if t.ttft_ms is not None:
                    ttfts.append(t.ttft_ms)
                if t.tpot_ms is not None:
                    tpots.append(t.tpot_ms)
            per_request.append({
                "request_id": item.request.request_id,
                "ttft_ms": t.ttft_ms,
                "tpot_ms": t.tpot_ms,
                "score": score,
                "success": t.success,
            })
        return {
            "ers": ers(scores),
            "n": len(items),
            "n_success": sum(1 for p in per_request if p["success"]),
            "per_request": per_request,
            "ttft_mean_ms": mean(ttfts) if ttfts else None,
            "tpot_mean_ms": mean(tpots) if tpots else None,
        }
