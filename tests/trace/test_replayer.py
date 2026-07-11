import pytest
from trace.loader import Request
from trace.client import StreamResult
from trace.replayer import Replayer, ReplayItem


class FakeClient:
    def __init__(self):
        self.order = []

    async def stream(self, body):
        self.order.append(body["messages"][0]["content"])
        return StreamResult(token_times_ms=[10.0, 20.0], text="ok", success=True)


@pytest.mark.asyncio
async def test_replay_returns_input_order():
    reqs = [
        Request(0, 0, {"messages": [{"role": "user", "content": "a"}]}),
        Request(1, 20, {"messages": [{"role": "user", "content": "b"}]}),
    ]
    client = FakeClient()
    # time_scale=0 -> no real waiting
    items = await Replayer(client, time_scale=0.0).run(reqs)
    assert [it.request.request_id for it in items] == [0, 1]
    assert all(isinstance(it, ReplayItem) for it in items)
    assert items[0].result.success is True


@pytest.mark.asyncio
async def test_replay_dispatches_by_timestamp_order():
    # Later-timestamp request scheduled later -> dispatched second even if listed first.
    reqs = [
        Request(1, 100, {"messages": [{"role": "user", "content": "late"}]}),
        Request(0, 0, {"messages": [{"role": "user", "content": "early"}]}),
    ]
    client = FakeClient()
    await Replayer(client, time_scale=0.001).run(reqs)  # 100ms -> 0.1ms scaled
    assert client.order[0] == "early"
