# THUA Plan 2 — GPQA wiring, Sweep, Docker packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the accuracy gate runnable end-to-end (real GPQA dataset + non-streaming client + combined Score), add a KV×scheduling sweep, and produce the submission Docker artifacts — all GPU-independent to build/test; the offline quantizer is scaffolded (recipe builder tested; execution runs later on GPU).

**Architecture:** Reuse Plan 1's `GpqaEvaluator`. Add a real dataset loader (row-mapping factored out so it is unit-testable without a download) and a non-streaming `ChatClient`. Add `run_gpqa` / `run_all` to the runner plus a combined report using `final_score`. Add a pure `expand_grid` for sweeps driven by a thin script. Add `model/` quantizer scaffolding whose recipe builder is tested but whose `.run()` imports `llmcompressor` lazily.

**Tech Stack:** Python 3.10+, `httpx`, `datasets` (GPQA, lazy import), `llmcompressor` (lazy import, GPU-only), `pytest`.

## Global Constraints

- Package layout is flat under `src/`: `config/`, `metrics/`, `tracing/`, `evaluation/`, `serve/`, `model/`, `runner.py`. No `thua.` prefix. Tests import bare (`from evaluation.gpqa import ...`).
- Score constants and `f(Δ)` are already in `metrics/`; reuse `final_score(ers_value, d)`.
- GPQA dataset is gated: `Idavidrein/gpqa`, config `gpqa_diamond`; needs `HF_TOKEN`. Never download in tests — inject rows / fake dataset.
- Accuracy eval uses `temperature=0` (greedy). Baseline `0.4`.
- `pytest` runs with `--import-mode=importlib`, `pythonpath=["src","."]`.
- Docker image must be self-contained (bake `/model`); compose keeps the fixed entrypoint + first 4 args (from Plan 1's `serve/`).

---

### Task 1: GPQA real dataset loader

**Files:**
- Create: `src/evaluation/gpqa_data.py`
- Test: `tests/evaluation/test_gpqa_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `map_row(raw: dict) -> dict` (maps a HF GPQA record to the `{Question, Correct Answer, Incorrect Answer 1..3}` shape `GpqaEvaluator` expects); `load_gpqa_rows(subset: str, n: int) -> list[dict]` (lazy-imports `datasets`, selects first `n`, applies `map_row`).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_gpqa_data.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_gpqa_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/gpqa_data.py
_FIELDS = ["Question", "Correct Answer",
          "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def map_row(raw: dict) -> dict:
    return {k: str(raw[k]).strip() for k in _FIELDS}


def load_gpqa_rows(subset: str, n: int) -> list[dict]:
    from datasets import load_dataset  # lazy: gated download, not needed for tests
    ds = load_dataset("Idavidrein/gpqa", subset)["train"]
    count = min(n, len(ds))
    return [map_row(ds[i]) for i in range(count)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_gpqa_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/gpqa_data.py tests/evaluation/test_gpqa_data.py
git commit -m "feat: GPQA dataset loader with unit-testable row mapping"
```

---

### Task 2: Non-streaming chat client

**Files:**
- Create: `src/evaluation/chat.py`
- Test: `tests/evaluation/test_chat.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class ChatClient` with `__init__(self, base_url: str, model: str, timeout_s: float, transport=None)` and `async def ask(self, prompt: str) -> str`. Sends a non-streaming `POST {base_url}/chat/completions` with a single user message and `temperature=0`, returns `choices[0].message.content` (empty string on any error).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_chat.py
import json
import httpx
import pytest
from evaluation.chat import ChatClient


@pytest.mark.asyncio
async def test_ask_returns_message_content():
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["stream"] is False
        payload = {"choices": [{"message": {"content": "Answer: B"}}]}
        return httpx.Response(200, json=payload)
    client = ChatClient("http://x/v1", "m", 10.0, transport=httpx.MockTransport(handler))
    out = await client.ask("q")
    assert out == "Answer: B"


@pytest.mark.asyncio
async def test_ask_error_returns_empty():
    def handler(request):
        return httpx.Response(500, json={})
    client = ChatClient("http://x/v1", "m", 10.0, transport=httpx.MockTransport(handler))
    assert await client.ask("q") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/chat.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_chat.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/chat.py tests/evaluation/test_chat.py
git commit -m "feat: non-streaming chat client for GPQA accuracy eval"
```

---

### Task 3: Combined final-score report + runner wiring

**Files:**
- Modify: `src/evaluation/report.py`
- Modify: `src/runner.py`
- Modify: `main.py`
- Test: `tests/evaluation/test_final_report.py`

**Interfaces:**
- Consumes: `final_score` (metrics/accuracy), `ChatClient` (Task 2), `load_gpqa_rows` (Task 1), `GpqaEvaluator` (Plan 1), `run_speed` (Plan 1).
- Produces: in `report.py` `format_final_report(speed: dict, acc: dict) -> str` (prints ERS, accuracy, Δ, `f(Δ)`, and `Score = 100·ERS·f(Δ)`); in `runner.py` `async def run_gpqa(cfg) -> dict` and `async def run_all(cfg) -> dict` returning `{"speed":..., "accuracy":..., "score": float}`; `main.py` gains subcommands `gpqa` and `all`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_final_report.py
from evaluation.report import format_final_report


def test_final_report_contains_score():
    speed = {"ers": 0.6, "n": 120, "n_success": 120,
             "ttft_mean_ms": 400.0, "tpot_mean_ms": 30.0}
    acc = {"accuracy": 0.32, "delta": 0.08}
    text = format_final_report(speed, acc)
    # Δ=0.08 -> f=1.0 -> Score = 100*0.6*1.0 = 60.00
    assert "Score" in text
    assert "60.00" in text
    assert "ERS" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_final_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_final_report'`

- [ ] **Step 3: Extend `report.py`**

Append to `src/evaluation/report.py`:

```python
from metrics.accuracy import penalty, final_score


def format_final_report(speed: dict, acc: dict) -> str:
    d = acc["delta"]
    score = final_score(speed["ers"], d)
    return "\n".join([
        format_speed_report(speed),
        "=== Accuracy ===",
        f"Accuracy:     {acc['accuracy']:.4f}",
        f"Delta:        {d:.4f}",
        f"f(delta):     {penalty(d):.4f}",
        "=== Final ===",
        f"Score:        {score:.2f}",
    ])
```

- [ ] **Step 4: Extend `runner.py`**

Append to `src/runner.py`:

```python
from evaluation.chat import ChatClient
from evaluation.gpqa import GpqaEvaluator
from evaluation.gpqa_data import load_gpqa_rows
from metrics.accuracy import final_score


async def run_gpqa(cfg) -> dict:
    rows = load_gpqa_rows(cfg.accuracy.subset, cfg.accuracy.n)
    client = ChatClient(cfg.accuracy.base_url, cfg.accuracy.model, cfg.accuracy.timeout_s)
    ev = GpqaEvaluator(rows, client.ask, baseline=cfg.accuracy.baseline)
    return await ev.evaluate()


async def run_all(cfg) -> dict:
    speed = await run_speed(cfg)
    accuracy = await run_gpqa(cfg)
    return {"speed": speed, "accuracy": accuracy,
            "score": final_score(speed["ers"], accuracy["delta"])}
```

- [ ] **Step 5: Extend `main.py`**

Add two subparsers and dispatch branches. In `_parse()` after the `compose` parser:

```python
    gp = sub.add_parser("gpqa", help="run GPQA accuracy and report delta")
    gp.add_argument("--config", required=True)
    gp.add_argument("--base-config", default="configs/base.yaml")
    ap = sub.add_parser("all", help="run speed + accuracy and report final Score")
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base.yaml")
```

In `main()` add branches (import `run_gpqa, run_all` and `format_final_report` at top):

```python
    elif args.cmd == "gpqa":
        acc = asyncio.run(run_gpqa(cfg))
        print(f"Accuracy: {acc['accuracy']:.4f}  Delta: {acc['delta']:.4f}")
    elif args.cmd == "all":
        result = asyncio.run(run_all(cfg))
        print(format_final_report(result["speed"], result["accuracy"]))
```

Update the import line in `main.py`:
```python
from runner import run_speed, gen_compose, run_gpqa, run_all
from evaluation.report import format_speed_report, format_final_report
```

- [ ] **Step 6: Run test + full suite**

Run: `pytest tests/evaluation/test_final_report.py -v`
Expected: PASS (1 passed)
Run: `pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/evaluation/report.py src/runner.py main.py tests/evaluation/test_final_report.py
git commit -m "feat: combined final-score report + gpqa/all runner and CLI"
```

---

### Task 4: KV×scheduling sweep

**Files:**
- Create: `src/sweep.py`
- Create: `scripts/sweep.py`
- Create: `configs/sweep/kv_sched.yaml`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: nothing (pure grid logic).
- Produces: `expand_grid(grid: dict[str, list]) -> list[dict]` — cartesian product of the grid, each result a flat dict `{dotted_key: value}` preserving grid key order; `apply_overrides(base: dict, flat: dict) -> dict` — applies dotted-key overrides onto a nested config dict (deep-copied).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep.py
from sweep import expand_grid, apply_overrides


def test_expand_grid_cartesian():
    grid = {
        "scheduling.max_num_batched_tokens": [4096, 8192],
        "kvcache.kv_cache_dtype": ["auto", "fp8_e4m3"],
    }
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert {"scheduling.max_num_batched_tokens": 4096,
            "kvcache.kv_cache_dtype": "auto"} in combos


def test_apply_overrides_nested():
    base = {"scheduling": {"max_num_batched_tokens": 1}, "kvcache": {"kv_cache_dtype": "x"}}
    out = apply_overrides(base, {"scheduling.max_num_batched_tokens": 4096,
                                 "kvcache.kv_cache_dtype": "fp8_e4m3"})
    assert out["scheduling"]["max_num_batched_tokens"] == 4096
    assert out["kvcache"]["kv_cache_dtype"] == "fp8_e4m3"
    # base not mutated
    assert base["scheduling"]["max_num_batched_tokens"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/sweep.py`**

```python
# src/sweep.py
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
```

- [ ] **Step 4: Write `configs/sweep/kv_sched.yaml` and `scripts/sweep.py`**

```yaml
# configs/sweep/kv_sched.yaml
grid:
  scheduling.max_num_batched_tokens: [4096, 8192, 16384]
  kvcache.kv_cache_dtype: [auto, fp8_e4m3]
  scheduling.scheduling_policy: [fcfs, priority]
```

```python
# scripts/sweep.py
"""Grid-search KV x scheduling against a running endpoint.

Usage: python scripts/sweep.py --config configs/experiment/exp_fp8.yaml \
       --grid configs/sweep/kv_sched.yaml
Prints one ERS row per combo. Requires an endpoint already serving (base_url in config).
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import yaml  # noqa: E402
from config.loader import deep_merge, load_experiment  # noqa: E402
from config.schema import (ServeConfig, KVCacheConfig, SchedulingConfig,  # noqa: E402
                           QuantConfig, BenchmarkConfig, AccuracyConfig, ExperimentConfig)
from sweep import expand_grid, apply_overrides  # noqa: E402
from runner import run_speed  # noqa: E402


def _to_cfg(merged: dict) -> ExperimentConfig:
    return ExperimentConfig(
        name=merged.get("name", "sweep"),
        serve=ServeConfig(**merged.get("serve", {})),
        kvcache=KVCacheConfig(**merged.get("kvcache", {})),
        scheduling=SchedulingConfig(**merged.get("scheduling", {})),
        quant=QuantConfig(**merged.get("quant", {})),
        benchmark=BenchmarkConfig(**merged.get("benchmark", {})),
        accuracy=AccuracyConfig(**merged.get("accuracy", {})),
    )


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--grid", required=True)
    args = ap.parse_args()

    base_cfg = load_experiment(args.config, base_path=args.base_config)
    base_dict = {
        "name": base_cfg.name, "serve": vars(base_cfg.serve), "kvcache": vars(base_cfg.kvcache),
        "scheduling": vars(base_cfg.scheduling), "quant": vars(base_cfg.quant),
        "benchmark": vars(base_cfg.benchmark), "accuracy": vars(base_cfg.accuracy),
    }
    with open(args.grid, encoding="utf-8") as fh:
        grid = yaml.safe_load(fh)["grid"]

    print("combo\tERS")
    for flat in expand_grid(grid):
        merged = apply_overrides(base_dict, flat)
        result = await run_speed(_to_cfg(merged))
        label = ",".join(f"{k.split('.')[-1]}={v}" for k, v in flat.items())
        print(f"{label}\t{result['ers']:.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/sweep.py scripts/sweep.py configs/sweep/kv_sched.yaml tests/test_sweep.py
git commit -m "feat: KV x scheduling sweep (grid expand + override apply + script)"
```

---

### Task 5: Offline quantizer scaffold

**Files:**
- Create: `src/model/__init__.py` (empty)
- Create: `src/model/base.py`
- Create: `src/model/fp8.py`
- Create: `src/model/registry.py`
- Create: `configs/quantize/fp8_w8a8.yaml`
- Create: `configs/quantize/awq_int4.yaml`
- Test: `tests/model/test_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class Quantizer(ABC)` with `recipe(self) -> list` and `run(self, src_model: str, out_dir: str) -> str`; `class Fp8Quantizer(Quantizer)` (`recipe()` returns an llm-compressor `QuantizationModifier` config as a plain dict — testable; `run()` lazy-imports `llmcompressor` and applies it, GPU-only); `get_quantizer(method: str) -> Quantizer` in `registry.py` (`fp8` → `Fp8Quantizer`, unknown → `ValueError`).

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_registry.py
import pytest
from model.registry import get_quantizer
from model.fp8 import Fp8Quantizer


def test_get_fp8_quantizer():
    q = get_quantizer("fp8")
    assert isinstance(q, Fp8Quantizer)


def test_unknown_raises():
    with pytest.raises(ValueError):
        get_quantizer("nope")


def test_fp8_recipe_targets_linear_fp8():
    recipe = Fp8Quantizer().recipe()
    assert recipe[0]["scheme"] == "FP8_DYNAMIC"
    assert "Linear" in recipe[0]["targets"]
    assert "lm_head" in recipe[0]["ignore"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `base.py`, `fp8.py`, `registry.py`**

```python
# src/model/base.py
from abc import ABC, abstractmethod


class Quantizer(ABC):
    @abstractmethod
    def recipe(self) -> list:
        ...

    @abstractmethod
    def run(self, src_model: str, out_dir: str) -> str:
        ...
```

```python
# src/model/fp8.py
from model.base import Quantizer


class Fp8Quantizer(Quantizer):
    def recipe(self) -> list:
        return [{
            "modifier": "QuantizationModifier",
            "targets": ["Linear"],
            "scheme": "FP8_DYNAMIC",
            "ignore": ["lm_head"],
        }]

    def run(self, src_model: str, out_dir: str) -> str:
        # GPU-only: lazy import so the module is importable without llmcompressor.
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifier = QuantizationModifier(
            targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=src_model, recipe=modifier, output_dir=out_dir)
        return out_dir
```

```python
# src/model/registry.py
from model.base import Quantizer
from model.fp8 import Fp8Quantizer

_QUANTIZERS = {"fp8": Fp8Quantizer}


def get_quantizer(method: str) -> Quantizer:
    if method not in _QUANTIZERS:
        raise ValueError(f"unknown quant method: {method} (have {list(_QUANTIZERS)})")
    return _QUANTIZERS[method]()
```

- [ ] **Step 4: Write the quantize config files**

```yaml
# configs/quantize/fp8_w8a8.yaml
quant:
  mode: offline
  method: fp8
```

```yaml
# configs/quantize/awq_int4.yaml
# Placeholder for AWQ INT4 (requires a calibration dataset; implement AwqQuantizer later).
quant:
  mode: offline
  method: awq
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/model/test_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/model/ configs/quantize/ tests/model/test_registry.py
git commit -m "feat: offline FP8 quantizer scaffold (testable recipe; lazy llmcompressor run)"
```

---

### Task 6: Submission Docker artifacts

**Files:**
- Create: `docker/Dockerfile`
- Create: `scripts/build_and_push.sh`
- Test: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a self-contained Dockerfile that bakes weights into `/model`, and a build/push helper. The test asserts structural invariants (base image, `/model`, no runtime HF download of weights).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docker_assets.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_bakes_model_and_uses_vllm_base():
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "vllm/vllm-openai" in text
    assert "/model" in text            # weights baked into image
    assert "COPY" in text or "ADD" in text


def test_build_script_pushes_public_image():
    text = (ROOT / "scripts" / "build_and_push.sh").read_text(encoding="utf-8")
    assert "docker build" in text
    assert "docker push" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docker_assets.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write `docker/Dockerfile`**

```dockerfile
# docker/Dockerfile — self-contained submission image.
# Build context expects a pre-downloaded (optionally pre-quantized) model at ./model.
FROM vllm/vllm-openai:v0.22.1

# Bake the model into the image so BTC's runner needs no network at startup.
# ./model is produced beforehand: HF download (fixed hash) and/or offline quantization.
COPY model /model

# vLLM's entrypoint is inherited from the base image; the submitted docker-compose.yml
# supplies the fixed entrypoint + args (see serve/compose.py output).
```

- [ ] **Step 4: Write `scripts/build_and_push.sh`**

```bash
#!/usr/bin/env bash
# Build and push the self-contained submission image to a PUBLIC Docker Hub repo.
# Usage: scripts/build_and_push.sh <dockerhub-user>/<repo>:<tag>
set -euo pipefail

IMAGE="${1:?usage: build_and_push.sh <user>/<repo>:<tag>}"

if [ ! -d "model" ]; then
  echo "ERROR: ./model not found. Download (fixed-hash) weights and/or run offline quantize first." >&2
  exit 1
fi

docker build -f docker/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
echo "Pushed $IMAGE (ensure the Docker Hub repo is PUBLIC before submitting)."
```

- [ ] **Step 5: Run test + full suite**

Run: `pytest tests/test_docker_assets.py -v`
Expected: PASS (2 passed)
Run: `pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile scripts/build_and_push.sh tests/test_docker_assets.py
git commit -m "feat: self-contained submission Dockerfile + build_and_push script"
```

---

## Notes for the implementer

- `datasets` and `llmcompressor` are lazy-imported inside functions; the suite does not require them installed. Install them only on the GPU box (`pip install datasets llmcompressor`) to actually run GPQA / offline quant.
- Real end-to-end accuracy: on the L4/MiG box after `docker compose up`, run `python main.py all --config configs/experiment/exp_fp8.yaml` (needs `HF_TOKEN` and a served endpoint).
- Real sweep: `python scripts/sweep.py --config configs/experiment/exp_fp8.yaml --grid configs/sweep/kv_sched.yaml`.
- AWQ INT4 (`AwqQuantizer`) needs a calibration set — deferred until a quality/latency need is proven (FP8 is expected to be safe on the accuracy gate).
