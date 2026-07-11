from pathlib import Path
from thua.trace.loader import load_trace, Request

FIXT = Path(__file__).parent / "fixtures" / "mini_trace.jsonl"


def test_load_count():
    reqs = load_trace(str(FIXT))
    assert len(reqs) == 2


def test_sorted_by_timestamp():
    reqs = load_trace(str(FIXT))
    assert [r.timestamp_ms for r in reqs] == [0, 50]
    assert reqs[0].request_id == 0


def test_body_preserved():
    reqs = load_trace(str(FIXT))
    assert reqs[0].body["max_tokens"] == 5
    assert isinstance(reqs[0], Request)
