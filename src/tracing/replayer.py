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
    def __init__(self, client, clock=time.perf_counter, sleep=asyncio.sleep, time_scale=1.0):
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.time_scale = time_scale

    async def run(self, requests: list[Request]) -> list[ReplayItem]:
        results: list[StreamResult | None] = [None] * len(requests)

        async def _one(idx: int, req: Request):
            delay = (req.timestamp_ms / 1000.0) * self.time_scale
            if delay > 0:
                await self.sleep(delay)
            results[idx] = await self.client.stream(req.body)

        await asyncio.gather(*(_one(i, r) for i, r in enumerate(requests)))
        return [ReplayItem(req, res) for req, res in zip(requests, results)]
