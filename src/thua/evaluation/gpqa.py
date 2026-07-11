import re

from thua.metrics.accuracy import delta

_TMPL = (
    "Answer the following multiple choice question. "
    "Think step by step, then end with 'Answer: X' where X is A, B, C or D.\n\n"
    "{q}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
)


def format_question(row: dict, rng) -> tuple[str, str]:
    opts = [row["Correct Answer"], row["Incorrect Answer 1"],
            row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    shuffled = [opts[i] for i in order]
    gold = "ABCD"[order.index(0)]
    prompt = _TMPL.format(q=row["Question"], a=shuffled[0], b=shuffled[1],
                          c=shuffled[2], d=shuffled[3])
    return prompt, gold


def extract_answer(text: str) -> str | None:
    m = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?([ABCD])\)?", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None


class GpqaEvaluator:
    def __init__(self, rows, chat_fn, baseline: float = 0.4):
        self.rows = list(rows)
        self.chat_fn = chat_fn
        self.baseline = baseline

    async def evaluate(self) -> dict:
        import random
        n_correct = 0
        for row in self.rows:
            rng = random.Random(row["Question"])
            prompt, gold = format_question(row, rng)
            text = await self.chat_fn(prompt)
            if extract_answer(text) == gold:
                n_correct += 1
        n = len(self.rows)
        acc = n_correct / n if n else 0.0
        return {"accuracy": acc, "n": n, "n_correct": n_correct,
                "delta": delta(self.baseline, acc)}
