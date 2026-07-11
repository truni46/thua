import json
from dataclasses import dataclass


@dataclass
class Request:
    request_id: int
    timestamp_ms: int
    body: dict


def load_trace(path: str) -> list[Request]:
    reqs: list[Request] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reqs.append(Request(
                request_id=row["request_id"],
                timestamp_ms=row["timestamp_ms"],
                body=row["body"],
            ))
    reqs.sort(key=lambda r: r.timestamp_ms)
    return reqs
