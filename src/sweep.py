import copy
import itertools


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def apply_overrides(base: dict, flat: dict) -> dict:
    out = copy.deepcopy(base)
    for dotted, value in flat.items():
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out
