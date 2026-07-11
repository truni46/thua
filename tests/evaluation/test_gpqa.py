import random
import pytest
from evaluation.gpqa import GpqaEvaluator, extract_answer, format_question

ROWS = [
    {"Question": "2+2?", "Correct Answer": "4",
     "Incorrect Answer 1": "3", "Incorrect Answer 2": "5", "Incorrect Answer 3": "6"},
    {"Question": "Sky color?", "Correct Answer": "blue",
     "Incorrect Answer 1": "red", "Incorrect Answer 2": "green", "Incorrect Answer 3": "pink"},
]


def test_extract_answer_variants():
    assert extract_answer("... therefore Answer: C") == "C"
    assert extract_answer("The answer is (B).") == "B"
    assert extract_answer("no letter here") is None


def test_format_question_gold_matches_correct():
    prompt, gold = format_question(ROWS[0], random.Random(0))
    assert gold in "ABCD"
    assert "2+2?" in prompt


@pytest.mark.asyncio
async def test_gpqa_all_correct():
    async def oracle(prompt):
        # cheat: the correct option text is always present; find its letter
        for letter in "ABCD":
            marker = f"{letter}) "
            idx = prompt.index(marker) + len(marker)
            line = prompt[idx:].splitlines()[0].strip()
            if line in ("4", "blue"):
                return f"Answer: {letter}"
        return "Answer: A"
    ev = GpqaEvaluator(ROWS, oracle, baseline=0.4)
    out = await ev.evaluate()
    assert out["n"] == 2
    assert out["n_correct"] == 2
    assert out["accuracy"] == 1.0
    assert abs(out["delta"] - (-0.6)) < 1e-9   # 0.4 - 1.0
