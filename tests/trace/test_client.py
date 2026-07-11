import json
import httpx
import pytest
from thua.trace.client import StreamingClient, StreamResult


def _sse(chunks: list[str]) -> bytes:
    lines = []
    for c in chunks:
        payload = {"choices": [{"delta": {"content": c}}]}
        lines.append(f"data: {json.dumps(payload)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@pytest.mark.asyncio
async def test_stream_collects_tokens():
    def handler(request):
        return httpx.Response(200, content=_sse(["Hel", "lo", "!"]))
    transport = httpx.MockTransport(handler)
    client = StreamingClient("http://x/v1", "m", 10.0, transport=transport)
    res = await client.stream({"messages": [{"role": "user", "content": "hi"}]})
    assert res.success is True
    assert res.text == "Hello!"
    assert len(res.token_times_ms) == 3
    assert res.token_times_ms == sorted(res.token_times_ms)  # monotonic


@pytest.mark.asyncio
async def test_stream_http_error_is_failure():
    def handler(request):
        return httpx.Response(500, content=b"boom")
    transport = httpx.MockTransport(handler)
    client = StreamingClient("http://x/v1", "m", 10.0, transport=transport)
    res = await client.stream({"messages": []})
    assert res.success is False
    assert res.error is not None
    assert res.token_times_ms == []
