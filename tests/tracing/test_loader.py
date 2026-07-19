from pathlib import Path
from tracing.loader import load_trace, Turn

FIXT = Path(__file__).parent / "fixtures"
TRACE = str(FIXT / "mini_trace.jsonl")
SPEC = str(FIXT / "mini_spec.json")


def test_load_count():
    turns = load_trace(TRACE, SPEC)
    assert len(turns) == 4


def test_turn_fields_and_output_pinned():
    turns = load_trace(TRACE, SPEC)
    assert isinstance(turns[0], Turn)
    assert turns[0].conv_id == 0
    assert turns[0].in_warmup is True
    assert turns[2].in_warmup is False
    assert turns[0].body["max_tokens"] == 4  # output_tokens_per_turn_pinned


def test_shared_system_prefix_identical_across_all_conversations():
    turns = load_trace(TRACE, SPEC)
    systems = {t.body["messages"][0]["content"] for t in turns}
    assert len(systems) == 1  # same shared prefix in every conversation
    assert turns[0].body["messages"][0]["role"] == "system"


def test_per_conversation_prefix_differs_between_conversations():
    turns = load_trace(TRACE, SPEC)
    conv0_t0 = turns[0].body["messages"][1]["content"]   # user turn-1 of conv 0
    conv1_t0 = turns[2].body["messages"][1]["content"]   # user turn-1 of conv 1
    assert conv0_t0 != conv1_t0


def test_context_accumulates_each_turn():
    turns = load_trace(TRACE, SPEC)
    # turn 1 = [system, user]; turn 2 = [system, user, assistant, user]
    assert len(turns[0].body["messages"]) == 2
    assert len(turns[1].body["messages"]) == 4
    assert turns[1].body["messages"][2]["role"] == "assistant"
    # turn 2's prompt extends turn 1's prompt (prefix reuse)
    assert turns[1].body["messages"][:2] == turns[0].body["messages"][:2]
