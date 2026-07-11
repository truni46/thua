_FIELDS = ["Question", "Correct Answer",
          "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def map_row(raw: dict) -> dict:
    return {k: str(raw[k]).strip() for k in _FIELDS}


def load_gpqa_rows(subset: str, n: int) -> list[dict]:
    from datasets import load_dataset  # lazy: gated download, not needed for tests
    ds = load_dataset("Idavidrein/gpqa", subset)["train"]
    count = min(n, len(ds))
    return [map_row(ds[i]) for i in range(count)]
