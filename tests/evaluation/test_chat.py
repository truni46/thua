import json
import httpx
import pytest
from evaluation.chat import ChatClient


@pytest.mark.asyncio
async def test_ask_returns_message_content():
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["stream"] is False
        payload = {"choices": [{"message": {"content": "Answer: B"}}]}
        return httpx.Response(200, json=payload)
    client = ChatClient("http://x/v1", "m", 10.0, transport=httpx.MockTransport(handler))
    out = await client.ask("q")
    assert out == "Answer: B"


@pytest.mark.asyncio
async def test_ask_error_returns_empty():
    def handler(request):
        return httpx.Response(500, json={})
    client = ChatClient("http://x/v1", "m", 10.0, transport=httpx.MockTransport(handler))
    assert await client.ask("q") == ""
