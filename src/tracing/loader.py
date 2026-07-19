import json
import random
from dataclasses import dataclass


@dataclass
class Turn:
    request_id: int
    conv_id: int
    turn_idx: int
    in_warmup: bool
    timestamp_ms: int
    think_ms: int
    body: dict


_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo consequat"
).split()


def _filler(n_tokens: int, seed: int) -> str:
    if n_tokens <= 0:
        return ""
    rnd = random.Random(seed)
    return " ".join(_WORDS[rnd.randrange(len(_WORDS))] for _ in range(n_tokens))


def load_trace(trace_path: str, spec_path: str = "grading-workload-spec.json") -> list[Turn]:
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    n_shared = int(spec["shared_system_prefix_tokens"])
    n_per_conv = int(spec["per_conversation_prefix_tokens"])
    n_user = int(spec["new_user_tokens_per_turn"])
    n_out = int(spec["output_tokens_per_turn_pinned"])
    shared = _filler(n_shared, seed=0)

    rows_by_conv: dict[int, list[dict]] = {}
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            rows_by_conv.setdefault(row["conv_id"], []).append(row)

    turns: list[Turn] = []
    rid = 0
    for cid in sorted(rows_by_conv):
        conv_rows = sorted(rows_by_conv[cid], key=lambda r: r["turn_idx"])
        per_conv = _filler(n_per_conv, seed=10_000 + cid)
        messages = [{"role": "system", "content": shared}]
        for i, row in enumerate(conv_rows):
            user = _filler(n_user, seed=1_000_000 + cid * 100 + row["turn_idx"])
            if i == 0:
                messages.append({"role": "user", "content": per_conv + " " + user})
            else:
                prev_out = _filler(n_out, seed=2_000_000 + cid * 100 + conv_rows[i - 1]["turn_idx"])
                messages.append({"role": "assistant", "content": prev_out})
                messages.append({"role": "user", "content": user})
            body = {
                "messages": [dict(m) for m in messages],
                "max_tokens": n_out,
                "temperature": 0.0,
            }
            turns.append(Turn(
                request_id=rid,
                conv_id=cid,
                turn_idx=int(row["turn_idx"]),
                in_warmup=bool(row.get("in_warmup", False)),
                timestamp_ms=int(row.get("timestamp_ms", 0)),
                think_ms=int(row.get("think_ms", 0)),
                body=body,
            ))
            rid += 1
    return turns
