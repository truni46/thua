import asyncio
import time
from dataclasses import dataclass

from tracing.client import StreamResult
from tracing.loader import Request


@dataclass
class ReplayItem:
    request: Request
    result: StreamResult


class Replayer:
    def __init__(self, client, clock=time.perf_counter, sleep=asyncio.sleep,
                 time_scale=1.0, on_result=None):
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.time_scale = time_scale
        self.on_result = on_result

    async def run(self, requests: list[Request]) -> list[ReplayItem]:
        results: list[StreamResult | None] = [None] * len(requests)

        async def _one(idx: int, req: Request):
            delay = (req.timestamp_ms / 1000.0) * self.time_scale
            if delay > 0:
                await self.sleep(delay)
            result = await self.client.stream(req.body)
            results[idx] = result
            if self.on_result is not None:
                self.on_result(idx, req, result)

        await asyncio.gather(*(_one(i, r) for i, r in enumerate(requests)))
        return [ReplayItem(req, res) for req, res in zip(requests, results)]
