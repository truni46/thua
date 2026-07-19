import httpx


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, transport=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport

    async def ask(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s,
                                         transport=self._transport) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return ""
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001 - eval client must not crash the run
            return ""
