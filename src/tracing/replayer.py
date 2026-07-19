import asyncio
import time
from dataclasses import dataclass

from tracing.client import StreamResult
from tracing.loader import Turn


@dataclass
class ReplayItem:
    turn: Turn
    result: StreamResult


class Replayer:
    def __init__(self, client, clock=time.perf_counter, sleep=asyncio.sleep,
                 time_scale=1.0, on_result=None):
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.time_scale = time_scale
        self.on_result = on_result

    async def run(self, turns: list[Turn]) -> list[ReplayItem]:
        results: dict[int, StreamResult] = {}
        convs: dict[int, list[Turn]] = {}
        for t in turns:
            convs.setdefault(t.conv_id, []).append(t)
        for seq in convs.values():
            seq.sort(key=lambda t: t.turn_idx)

        async def run_conv(seq: list[Turn]):
            start_delay = (seq[0].timestamp_ms / 1000.0) * self.time_scale
            if start_delay > 0:
                await self.sleep(start_delay)
            for i, t in enumerate(seq):
                res = await self.client.stream(t.body)
                results[t.request_id] = res
                if self.on_result is not None:
                    self.on_result(t.request_id, t, res)
                if i < len(seq) - 1 and t.think_ms > 0:
                    await self.sleep((t.think_ms / 1000.0) * self.time_scale)

        await asyncio.gather(*(run_conv(seq) for seq in convs.values()))
        return [ReplayItem(t, results[t.request_id]) for t in turns]
