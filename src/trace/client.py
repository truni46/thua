import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class StreamResult:
    token_times_ms: list[float] = field(default_factory=list)
    text: str = ""
    success: bool = False
    error: str | None = None


class StreamingClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, transport=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport

    async def stream(self, body: dict) -> StreamResult:
        payload = dict(body)
        payload["model"] = self.model
        payload["stream"] = True
        url = f"{self.base_url}/chat/completions"
        res = StreamResult()
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s,
                                         transport=self._transport) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        res.error = f"HTTP {resp.status_code}"
                        return res
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            res.token_times_ms.append((time.perf_counter() - start) * 1000.0)
                            res.text += content
            res.success = len(res.token_times_ms) > 0
            if not res.success and res.error is None:
                res.error = "no tokens"
        except Exception as exc:  # noqa: BLE001 - benchmark client must not crash the run
            res.error = repr(exc)
        return res
