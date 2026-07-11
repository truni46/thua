import pytest
from thua.trace.loader import Request
from thua.trace.client import StreamResult
from thua.trace.replayer import Replayer
from thua.metrics.scoring import ScoringConfig
from thua.evaluation.speed import SpeedEvaluator


class FakeClient:
    def __init__(self, results):
        self._results = results
        self._i = 0

    async def stream(self, body):
        r = self._results[self._i]
        self._i += 1
        return r


@pytest.mark.asyncio
async def test_speed_evaluator_computes_ers():
    reqs = [
        Request(0, 0, {"messages": [{"role": "user", "content": "a"}]}),
        Request(1, 0, {"messages": [{"role": "user", "content": "b"}]}),
    ]
    # req0: ttft=100ms, then tokens at +20ms each -> best case; req1: failure
    results = [
        StreamResult(token_times_ms=[100.0, 120.0, 140.0], text="x", success=True),
        StreamResult(token_times_ms=[], text="", success=False, error="timeout"),
    ]
    ev = SpeedEvaluator(reqs, Replayer(FakeClient(results), time_scale=0.0), ScoringConfig())
    out = await ev.evaluate()
    assert out["n"] == 2
    assert out["n_success"] == 1
    # req0 score ~1.0, req1 score 0 -> ERS ~0.5
    assert abs(out["ers"] - 0.5) < 1e-6
    assert out["per_request"][1]["success"] is False
