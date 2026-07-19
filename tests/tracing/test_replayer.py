import pytest
from tracing.loader import Turn
from tracing.client import StreamResult
from tracing.replayer import Replayer, ReplayItem


def _turn(rid, cid, tidx, ts, content, warmup=False, think=0):
    return Turn(request_id=rid, conv_id=cid, turn_idx=tidx, in_warmup=warmup,
                timestamp_ms=ts, think_ms=think,
                body={"messages": [{"role": "user", "content": content}]})


class FakeClient:
    def __init__(self):
        self.order = []

    async def stream(self, body):
        self.order.append(body["messages"][-1]["content"])
        return StreamResult(token_times_ms=[10.0, 20.0], text="ok", success=True)


@pytest.mark.asyncio
async def test_replay_returns_input_order():
    turns = [_turn(0, 0, 0, 0, "a"), _turn(1, 1, 0, 0, "b")]
    items = await Replayer(FakeClient(), time_scale=0.0).run(turns)
    assert [it.turn.request_id for it in items] == [0, 1]
    assert all(isinstance(it, ReplayItem) for it in items)
    assert items[0].result.success is True


@pytest.mark.asyncio
async def test_replay_serializes_turns_within_conversation():
    turns = [_turn(1, 0, 1, 0, "second", think=0),
             _turn(0, 0, 0, 0, "first", think=0)]
    client = FakeClient()
    await Replayer(client, time_scale=0.0).run(turns)
    assert client.order == ["first", "second"]


@pytest.mark.asyncio
async def test_replay_invokes_on_result_per_turn():
    turns = [_turn(0, 0, 0, 0, "a"), _turn(1, 1, 0, 0, "b")]
    seen = []

    def cb(rid, turn, result):
        seen.append((rid, turn.conv_id, result.success))

    await Replayer(FakeClient(), time_scale=0.0, on_result=cb).run(turns)
    assert sorted(seen) == [(0, 0, True), (1, 1, True)]
