from evaluation.gpqa_data import map_row

RAW = {
    "Question": "Q?",
    "Correct Answer": "right",
    "Incorrect Answer 1": "w1",
    "Incorrect Answer 2": "w2",
    "Incorrect Answer 3": "w3",
    "Extra": "ignored",
}


def test_map_row_keeps_required_fields():
    out = map_row(RAW)
    assert out == {
        "Question": "Q?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
    }


def test_map_row_strips_whitespace():
    out = map_row({**RAW, "Question": "  Q?  ", "Correct Answer": " right "})
    assert out["Question"] == "Q?"
    assert out["Correct Answer"] == "right"
