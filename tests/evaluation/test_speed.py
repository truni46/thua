import pytest
from tracing.loader import Turn
from tracing.client import StreamResult
from tracing.replayer import Replayer
from metrics import ScoringConfig
from evaluation.speed import SpeedEvaluator


def _turn(rid, cid, content, warmup):
    return Turn(request_id=rid, conv_id=cid, turn_idx=0, in_warmup=warmup,
                timestamp_ms=0, think_ms=0,
                body={"messages": [{"role": "user", "content": content}]})


class FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping

    async def stream(self, body):
        return self.mapping[body["messages"][-1]["content"]]


@pytest.mark.asyncio
async def test_speed_evaluator_excludes_warmup_and_scores_rest():
    turns = [_turn(0, 0, "warm", warmup=True), _turn(1, 1, "good", warmup=False)]
    results = {
        "warm": StreamResult(token_times_ms=[999.0], text="x", success=True),
        "good": StreamResult(token_times_ms=[10.0, 11.0], text="y", success=True),
    }
    ev = SpeedEvaluator(turns, Replayer(FakeClient(results), time_scale=0.0), ScoringConfig())
    out = await ev.evaluate()
    assert out["n"] == 1
    assert out["n_total"] == 2
    assert out["n_success"] == 1
    # scored request: ttft=10 (floor), tpot=1 (floor) -> score 1.0 -> ERS 1.0
    assert abs(out["ers"] - 1.0) < 1e-6
    warm_row = next(p for p in out["per_request"] if p["in_warmup"])
    assert warm_row["score"] is None


@pytest.mark.asyncio
async def test_failure_scores_zero():
    turns = [_turn(0, 0, "bad", warmup=False)]
    results = {"bad": StreamResult(token_times_ms=[], text="", success=False, error="timeout")}
    ev = SpeedEvaluator(turns, Replayer(FakeClient(results), time_scale=0.0), ScoringConfig())
    out = await ev.evaluate()
    assert out["n"] == 1
    assert out["n_success"] == 0
    assert out["ers"] == 0.0
