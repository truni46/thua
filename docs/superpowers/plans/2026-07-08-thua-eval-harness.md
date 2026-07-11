# THUA Evaluation Harness Implementation Plan

> **⚠️ Path note (post-implementation):** This plan was written against a `src/thua/` package
> layout with a `trace/` module. The delivered code uses a **flat** layout under `src/`
> (imports are bare, e.g. `from metrics.scoring import ...`, `from runner import ...`) and the
> `trace/` module was renamed `tracing/` to avoid shadowing Python's stdlib `trace`. When
> reading tasks below, mentally map `src/thua/X` → `src/X`, `from thua.X` → `from X`, and
> `trace/` → `tracing/`. Everything else (steps, tests, code) is accurate.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU-independent core of the THUA pipeline — config, scoring/accuracy metrics, trace replay, speed + GPQA evaluators, docker-compose generation, and a `main.py` CLI — so serving configs can be scored against any OpenAI-compatible endpoint.

**Architecture:** Pure-function metrics at the core (ERS, f(Δ), TTFT/TPOT) with exact constants from the competition rules. An async `httpx` streaming client replays the trace honoring `timestamp_ms` and captures per-token timings. Evaluators (`SpeedEvaluator`, `GpqaEvaluator`) combine client + metrics into a report. A pluggable `ServeBackend` emits `docker-compose.yml`. Everything is config-driven (plain YAML + dataclasses).

**Tech Stack:** Python 3.10+, `httpx` (async streaming), `pyyaml`, `datasets` (GPQA, injected so tests don't download), `pytest` + `pytest-asyncio`.

## Global Constraints

- Scoring constants (verbatim): `F_ttft=100 ms`, `C_ttft=1500 ms`, `F_tpot=20 ms`, `C_tpot=45 ms`, `γ=2`, `w=0.5`.
- Accuracy penalty: `f(Δ)=1.0 if Δ≤0.10`; `f(Δ)=1.0-(Δ-0.10)/0.06 if 0.10<Δ<0.16`; `f(Δ)=0.0 if Δ≥0.16`. Baseline accuracy `=0.4`.
- A request scores `0` on error / timeout / 0 tokens.
- `Score = 100 · ERS · f(Δ)`.
- Config style: plain YAML + dataclasses, manual `base + override` deep-merge. No Hydra, no pydantic.
- vLLM entrypoint + first 4 args (`--model`, `--served-model-name`, `--host`, `--port`) are fixed by competition rule; generated compose must keep them.
- Python package root: `src/thua/`. Tests under `tests/`. Line length ≤ 100.

---

### Task 1: Project scaffold + ERS scoring

**Files:**
- Create: `pyproject.toml`
- Create: `src/thua/__init__.py` (empty)
- Create: `src/thua/metrics/__init__.py` (empty)
- Create: `src/thua/metrics/scoring.py`
- Test: `tests/metrics/test_scoring.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ScoringConfig` dataclass (`f_ttft, c_ttft, f_tpot, c_tpot, gamma, w`); `s_ttft(ttft_ms: float, cfg) -> float`; `s_tpot(tpot_ms: float | None, cfg) -> float`; `request_score(ttft_ms: float | None, tpot_ms: float | None, success: bool, cfg) -> float`; `ers(scores: list[float]) -> float`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "thua"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["httpx>=0.27", "pyyaml>=6.0"]

[project.optional-dependencies]
gpqa = ["datasets>=2.18"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/metrics/test_scoring.py
import math
from thua.metrics.scoring import ScoringConfig, s_ttft, s_tpot, request_score, ers

CFG = ScoringConfig()

def test_s_ttft_at_floor_is_one():
    assert s_ttft(100, CFG) == 1.0

def test_s_ttft_at_ceiling_is_zero():
    assert s_ttft(1500, CFG) == 0.0

def test_s_ttft_midpoint_gamma_squared():
    # (1500-800)/1400 = 0.5 -> 0.25 with gamma=2
    assert math.isclose(s_ttft(800, CFG), 0.25, rel_tol=1e-9)

def test_s_tpot_none_is_zero():
    assert s_tpot(None, CFG) == 0.0

def test_s_tpot_midpoint():
    # (45-32.5)/25 = 0.5 -> 0.25
    assert math.isclose(s_tpot(32.5, CFG), 0.25, rel_tol=1e-9)

def test_request_score_best_case():
    assert math.isclose(request_score(100, 20, True, CFG), 1.0, rel_tol=1e-9)

def test_request_score_failure_is_zero():
    assert request_score(120, 25, False, CFG) == 0.0

def test_request_score_none_ttft_is_zero():
    assert request_score(None, 25, True, CFG) == 0.0

def test_ers_average():
    assert math.isclose(ers([1.0, 0.0, 0.5]), 0.5, rel_tol=1e-9)

def test_ers_empty_is_zero():
    assert ers([]) == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/metrics/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'thua.metrics.scoring'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/thua/metrics/scoring.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    f_ttft: float = 100.0
    c_ttft: float = 1500.0
    f_tpot: float = 20.0
    c_tpot: float = 45.0
    gamma: float = 2.0
    w: float = 0.5


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def s_ttft(ttft_ms: float, cfg: ScoringConfig) -> float:
    x = _clamp01((cfg.c_ttft - ttft_ms) / (cfg.c_ttft - cfg.f_ttft))
    return x ** cfg.gamma


def s_tpot(tpot_ms: float | None, cfg: ScoringConfig) -> float:
    if tpot_ms is None:
        return 0.0
    x = _clamp01((cfg.c_tpot - tpot_ms) / (cfg.c_tpot - cfg.f_tpot))
    return x ** cfg.gamma


def request_score(ttft_ms: float | None, tpot_ms: float | None,
                  success: bool, cfg: ScoringConfig) -> float:
    if not success or ttft_ms is None:
        return 0.0
    return cfg.w * s_ttft(ttft_ms, cfg) + (1.0 - cfg.w) * s_tpot(tpot_ms, cfg)


def ers(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/metrics/test_scoring.py -v`
Expected: PASS (10 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/thua/__init__.py src/thua/metrics/ tests/metrics/test_scoring.py
git commit -m "feat: project scaffold + ERS scoring metrics"
```

---

### Task 2: Accuracy penalty & final score

**Files:**
- Create: `src/thua/metrics/accuracy.py`
- Test: `tests/metrics/test_accuracy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `delta(baseline: float, acc: float) -> float`; `penalty(d: float) -> float`; `final_score(ers_value: float, d: float) -> float`; constant `DEFAULT_BASELINE = 0.4`.

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_accuracy.py
import math
from thua.metrics.accuracy import delta, penalty, final_score, DEFAULT_BASELINE

def test_default_baseline():
    assert DEFAULT_BASELINE == 0.4

def test_delta():
    assert math.isclose(delta(0.4, 0.32), 0.08, rel_tol=1e-9)

def test_penalty_no_drop():
    assert penalty(0.10) == 1.0
    assert penalty(0.05) == 1.0

def test_penalty_full_cutoff():
    assert penalty(0.16) == 0.0
    assert penalty(0.20) == 0.0

def test_penalty_linear_midpoint():
    # Δ=0.13 -> 1 - (0.03/0.06) = 0.5
    assert math.isclose(penalty(0.13), 0.5, rel_tol=1e-9)

def test_final_score():
    # ERS 0.6, Δ 0.13 -> 100*0.6*0.5 = 30
    assert math.isclose(final_score(0.6, 0.13), 30.0, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/metrics/test_accuracy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/metrics/accuracy.py
DEFAULT_BASELINE = 0.4


def delta(baseline: float, acc: float) -> float:
    return baseline - acc


def penalty(d: float) -> float:
    if d <= 0.10:
        return 1.0
    if d >= 0.16:
        return 0.0
    return 1.0 - (d - 0.10) / 0.06


def final_score(ers_value: float, d: float) -> float:
    return 100.0 * ers_value * penalty(d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/metrics/test_accuracy.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/metrics/accuracy.py tests/metrics/test_accuracy.py
git commit -m "feat: accuracy penalty f(delta) + final score"
```

---

### Task 3: Latency (TTFT / TPOT) from token timestamps

**Files:**
- Create: `src/thua/metrics/latency.py`
- Test: `tests/metrics/test_latency.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `@dataclass RequestTiming(ttft_ms: float | None, tpot_ms: float | None, n_tokens: int, success: bool)`; `compute_timing(token_times_ms: list[float], success: bool) -> RequestTiming`. `token_times_ms` are wall-clock ms of each output token **relative to request send**.

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_latency.py
import math
from thua.metrics.latency import compute_timing, RequestTiming

def test_ttft_is_first_token_time():
    t = compute_timing([120.0, 145.0, 170.0], True)
    assert t.ttft_ms == 120.0

def test_tpot_is_mean_interval():
    # (170-120)/(3-1) = 25
    t = compute_timing([120.0, 145.0, 170.0], True)
    assert math.isclose(t.tpot_ms, 25.0, rel_tol=1e-9)
    assert t.n_tokens == 3

def test_single_token_has_no_tpot():
    t = compute_timing([120.0], True)
    assert t.ttft_ms == 120.0
    assert t.tpot_ms is None

def test_failure_is_all_none():
    t = compute_timing([120.0, 130.0], False)
    assert t == RequestTiming(None, None, 0, False)

def test_empty_tokens_is_failure_shape():
    t = compute_timing([], True)
    assert t.success is False
    assert t.ttft_ms is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/metrics/test_latency.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/metrics/latency.py
from dataclasses import dataclass


@dataclass
class RequestTiming:
    ttft_ms: float | None
    tpot_ms: float | None
    n_tokens: int
    success: bool


def compute_timing(token_times_ms: list[float], success: bool) -> RequestTiming:
    if not success or not token_times_ms:
        return RequestTiming(None, None, 0, False)
    ttft = token_times_ms[0]
    n = len(token_times_ms)
    tpot = (token_times_ms[-1] - token_times_ms[0]) / (n - 1) if n > 1 else None
    return RequestTiming(ttft, tpot, n, True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/metrics/test_latency.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/metrics/latency.py tests/metrics/test_latency.py
git commit -m "feat: TTFT/TPOT latency computation from token timestamps"
```

---

### Task 4: Config schema + loader (base + override merge)

**Files:**
- Create: `src/thua/config/__init__.py` (empty)
- Create: `src/thua/config/schema.py`
- Create: `src/thua/config/loader.py`
- Create: `configs/base.yaml`
- Create: `configs/experiment/exp_fp8.yaml`
- Test: `tests/config/test_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces dataclasses in `schema.py`: `ServeConfig(backend, model_path, served_model_name, host, port, max_model_len, gpu_memory_utilization, dtype, extra_args: dict)`, `KVCacheConfig(enable_prefix_caching, kv_cache_dtype, block_size, cpu_offload_gb)`, `SchedulingConfig(enable_chunked_prefill, max_num_batched_tokens, long_prefill_token_threshold, max_num_seqs, max_num_partial_prefills, scheduling_policy, preemption_mode)`, `QuantConfig(mode, method)`, `BenchmarkConfig(trace_path, base_url, model, timeout_s)`, `AccuracyConfig(dataset, subset, n, baseline, base_url, model, timeout_s)`, `ExperimentConfig(name, serve, kvcache, scheduling, quant, benchmark, accuracy)`.
- Produces in `loader.py`: `deep_merge(base: dict, override: dict) -> dict`; `load_experiment(path: str, base_path: str = "configs/base.yaml") -> ExperimentConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_loader.py
from thua.config.loader import deep_merge, load_experiment

def test_deep_merge_overrides_nested():
    base = {"serve": {"port": 8000, "host": "0.0.0.0"}, "x": 1}
    over = {"serve": {"port": 9000}, "y": 2}
    out = deep_merge(base, over)
    assert out == {"serve": {"port": 9000, "host": "0.0.0.0"}, "x": 1, "y": 2}

def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}

def test_load_experiment_builds_typed_config(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "serve: {backend: vllm, model_path: /model, served_model_name: Qwen3.5-2B, "
        "host: 0.0.0.0, port: 8000, max_model_len: 45000, gpu_memory_utilization: 0.9, "
        "dtype: auto, extra_args: {}}\n"
        "kvcache: {enable_prefix_caching: true, kv_cache_dtype: auto, block_size: 16, "
        "cpu_offload_gb: 0}\n"
        "scheduling: {enable_chunked_prefill: true, max_num_batched_tokens: 8192, "
        "long_prefill_token_threshold: 0, max_num_seqs: 32, max_num_partial_prefills: 1, "
        "scheduling_policy: fcfs, preemption_mode: recompute}\n"
        "quant: {mode: online, method: fp8}\n"
        "benchmark: {trace_path: trace-round1.jsonl, base_url: http://localhost:8000/v1, "
        "model: Qwen3.5-2B, timeout_s: 60}\n"
        "accuracy: {dataset: Idavidrein/gpqa, subset: gpqa_diamond, n: 198, baseline: 0.4, "
        "base_url: http://localhost:8000/v1, model: Qwen3.5-2B, timeout_s: 120}\n",
        encoding="utf-8",
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text("name: test_fp8\nserve: {port: 9000}\n", encoding="utf-8")
    cfg = load_experiment(str(exp), base_path=str(base))
    assert cfg.name == "test_fp8"
    assert cfg.serve.port == 9000          # override applied
    assert cfg.serve.max_model_len == 45000  # inherited from base
    assert cfg.kvcache.enable_prefix_caching is True
    assert cfg.scheduling.max_num_batched_tokens == 8192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `schema.py`**

```python
# src/thua/config/schema.py
from dataclasses import dataclass, field


@dataclass
class ServeConfig:
    backend: str = "vllm"
    model_path: str = "/model"
    served_model_name: str = "Qwen3.5-2B"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 45000
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    extra_args: dict = field(default_factory=dict)


@dataclass
class KVCacheConfig:
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"
    block_size: int = 16
    cpu_offload_gb: float = 0.0


@dataclass
class SchedulingConfig:
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    long_prefill_token_threshold: int = 0
    max_num_seqs: int = 32
    max_num_partial_prefills: int = 1
    scheduling_policy: str = "fcfs"
    preemption_mode: str = "recompute"


@dataclass
class QuantConfig:
    mode: str = "online"   # online | offline | none
    method: str = "fp8"    # fp8 | awq | gptq | none


@dataclass
class BenchmarkConfig:
    trace_path: str = "trace-round1.jsonl"
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3.5-2B"
    timeout_s: float = 60.0


@dataclass
class AccuracyConfig:
    dataset: str = "Idavidrein/gpqa"
    subset: str = "gpqa_diamond"
    n: int = 198
    baseline: float = 0.4
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3.5-2B"
    timeout_s: float = 120.0


@dataclass
class ExperimentConfig:
    name: str
    serve: ServeConfig
    kvcache: KVCacheConfig
    scheduling: SchedulingConfig
    quant: QuantConfig
    benchmark: BenchmarkConfig
    accuracy: AccuracyConfig
```

- [ ] **Step 4: Write `loader.py`**

```python
# src/thua/config/loader.py
import copy
import yaml

from thua.config.schema import (
    ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig,
    BenchmarkConfig, AccuracyConfig, ExperimentConfig,
)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_experiment(path: str, base_path: str = "configs/base.yaml") -> ExperimentConfig:
    merged = deep_merge(_load_yaml(base_path), _load_yaml(path))
    return ExperimentConfig(
        name=merged.get("name", "unnamed"),
        serve=ServeConfig(**merged.get("serve", {})),
        kvcache=KVCacheConfig(**merged.get("kvcache", {})),
        scheduling=SchedulingConfig(**merged.get("scheduling", {})),
        quant=QuantConfig(**merged.get("quant", {})),
        benchmark=BenchmarkConfig(**merged.get("benchmark", {})),
        accuracy=AccuracyConfig(**merged.get("accuracy", {})),
    )
```

- [ ] **Step 5: Write the real `configs/base.yaml` and `configs/experiment/exp_fp8.yaml`**

```yaml
# configs/base.yaml
serve:
  backend: vllm
  model_path: /model
  served_model_name: Qwen3.5-2B
  host: 0.0.0.0
  port: 8000
  max_model_len: 45000
  gpu_memory_utilization: 0.9
  dtype: auto
  extra_args: {}
kvcache:
  enable_prefix_caching: true
  kv_cache_dtype: auto
  block_size: 16
  cpu_offload_gb: 0
scheduling:
  enable_chunked_prefill: true
  max_num_batched_tokens: 8192
  long_prefill_token_threshold: 0
  max_num_seqs: 32
  max_num_partial_prefills: 1
  scheduling_policy: fcfs
  preemption_mode: recompute
quant:
  mode: online
  method: fp8
benchmark:
  trace_path: trace-round1.jsonl
  base_url: http://localhost:8000/v1
  model: Qwen3.5-2B
  timeout_s: 60
accuracy:
  dataset: Idavidrein/gpqa
  subset: gpqa_diamond
  n: 198
  baseline: 0.4
  base_url: http://localhost:8000/v1
  model: Qwen3.5-2B
  timeout_s: 120
```

```yaml
# configs/experiment/exp_fp8.yaml
name: fp8_kv_chunked
quant:
  mode: online
  method: fp8
kvcache:
  kv_cache_dtype: fp8_e4m3
scheduling:
  max_num_batched_tokens: 8192
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/config/test_loader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add src/thua/config/ configs/base.yaml configs/experiment/exp_fp8.yaml tests/config/test_loader.py
git commit -m "feat: config schema + base/override YAML loader"
```

---

### Task 5: Trace loader

**Files:**
- Create: `src/thua/trace/__init__.py` (empty)
- Create: `src/thua/trace/loader.py`
- Create: `tests/trace/fixtures/mini_trace.jsonl`
- Test: `tests/trace/test_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `@dataclass Request(request_id: int, timestamp_ms: int, body: dict)`; `load_trace(path: str) -> list[Request]` (sorted by `timestamp_ms`).

- [ ] **Step 1: Create the fixture**

```json
{"request_id": 2, "timestamp_ms": 50, "workload_type": "conversation", "body": {"model": "Qwen3.5-2B", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0, "seed": 42}}
{"request_id": 0, "timestamp_ms": 0, "workload_type": "conversation", "body": {"model": "Qwen3.5-2B", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5, "temperature": 0, "seed": 42}}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/trace/test_loader.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/trace/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

```python
# src/thua/trace/loader.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/trace/test_loader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/thua/trace/__init__.py src/thua/trace/loader.py tests/trace/
git commit -m "feat: trace loader (jsonl -> Request, sorted by timestamp)"
```

---

### Task 6: Async streaming client

**Files:**
- Create: `src/thua/trace/client.py`
- Test: `tests/trace/test_client.py`

**Interfaces:**
- Consumes: `Request` (not required — takes a raw body dict).
- Produces: `@dataclass StreamResult(token_times_ms: list[float], text: str, success: bool, error: str | None)`; `class StreamingClient` with `__init__(self, base_url: str, model: str, timeout_s: float, transport=None)` and `async def stream(self, body: dict) -> StreamResult`. `token_times_ms` are `perf_counter`-based ms relative to the moment the request is sent. Streaming uses OpenAI-compatible `POST {base_url}/chat/completions` with `stream=True`; each SSE `data:` chunk carrying a non-empty `choices[0].delta.content` records one token timestamp. `[DONE]` ends the stream. The optional `transport` is passed to `httpx.AsyncClient` for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/trace/test_client.py
import json
import httpx
import pytest
from thua.trace.client import StreamingClient, StreamResult

def _sse(chunks: list[str]) -> bytes:
    lines = []
    for c in chunks:
        payload = {"choices": [{"delta": {"content": c}}]}
        lines.append(f"data: {json.dumps(payload)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()

@pytest.mark.asyncio
async def test_stream_collects_tokens():
    def handler(request):
        return httpx.Response(200, content=_sse(["Hel", "lo", "!"]))
    transport = httpx.MockTransport(handler)
    client = StreamingClient("http://x/v1", "m", 10.0, transport=transport)
    res = await client.stream({"messages": [{"role": "user", "content": "hi"}]})
    assert res.success is True
    assert res.text == "Hello!"
    assert len(res.token_times_ms) == 3
    assert res.token_times_ms == sorted(res.token_times_ms)  # monotonic

@pytest.mark.asyncio
async def test_stream_http_error_is_failure():
    def handler(request):
        return httpx.Response(500, content=b"boom")
    transport = httpx.MockTransport(handler)
    client = StreamingClient("http://x/v1", "m", 10.0, transport=transport)
    res = await client.stream({"messages": []})
    assert res.success is False
    assert res.error is not None
    assert res.token_times_ms == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trace/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/trace/client.py
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class StreamResult:
    token_times_ms: list[float] = field(default_factory=list)
    text: str = ""
    success: bool = False
    error: str | None = None


class StreamingClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, transport=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport

    async def stream(self, body: dict) -> StreamResult:
        payload = dict(body)
        payload["model"] = self.model
        payload["stream"] = True
        url = f"{self.base_url}/chat/completions"
        res = StreamResult()
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s,
                                         transport=self._transport) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        res.error = f"HTTP {resp.status_code}"
                        return res
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            res.token_times_ms.append((time.perf_counter() - start) * 1000.0)
                            res.text += content
            res.success = len(res.token_times_ms) > 0
            if not res.success and res.error is None:
                res.error = "no tokens"
        except Exception as exc:  # noqa: BLE001 - benchmark client must not crash the run
            res.error = repr(exc)
        return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trace/test_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/trace/client.py tests/trace/test_client.py
git commit -m "feat: async streaming OpenAI-compatible client with per-token timing"
```

---

### Task 7: Trace replayer (honor timestamps)

**Files:**
- Create: `src/thua/trace/replayer.py`
- Test: `tests/trace/test_replayer.py`

**Interfaces:**
- Consumes: `Request` (Task 5), `StreamResult` (Task 6).
- Produces: `@dataclass ReplayItem(request: Request, result: StreamResult)`; `class Replayer` with `__init__(self, client, clock=time.perf_counter, sleep=asyncio.sleep, time_scale=1.0)` and `async def run(self, requests: list[Request]) -> list[ReplayItem]`. The client only needs an `async stream(body) -> StreamResult` method (duck-typed, so tests pass a fake). Each request is dispatched at `timestamp_ms/1000 * time_scale` seconds after `run` starts; all run concurrently via `asyncio.gather`. Results are returned in the input order.

- [ ] **Step 1: Write the failing test**

```python
# tests/trace/test_replayer.py
import asyncio
import pytest
from thua.trace.loader import Request
from thua.trace.client import StreamResult
from thua.trace.replayer import Replayer, ReplayItem

class FakeClient:
    def __init__(self):
        self.order = []
    async def stream(self, body):
        self.order.append(body["messages"][0]["content"])
        return StreamResult(token_times_ms=[10.0, 20.0], text="ok", success=True)

@pytest.mark.asyncio
async def test_replay_returns_input_order():
    reqs = [
        Request(0, 0, {"messages": [{"role": "user", "content": "a"}]}),
        Request(1, 20, {"messages": [{"role": "user", "content": "b"}]}),
    ]
    client = FakeClient()
    # time_scale=0 -> no real waiting
    items = await Replayer(client, time_scale=0.0).run(reqs)
    assert [it.request.request_id for it in items] == [0, 1]
    assert all(isinstance(it, ReplayItem) for it in items)
    assert items[0].result.success is True

@pytest.mark.asyncio
async def test_replay_dispatches_by_timestamp_order():
    # Later-timestamp request scheduled later -> dispatched second even if listed first.
    reqs = [
        Request(1, 100, {"messages": [{"role": "user", "content": "late"}]}),
        Request(0, 0, {"messages": [{"role": "user", "content": "early"}]}),
    ]
    client = FakeClient()
    await Replayer(client, time_scale=0.001).run(reqs)  # 100ms -> 0.1ms scaled
    assert client.order[0] == "early"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trace/test_replayer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/trace/replayer.py
import asyncio
import time
from dataclasses import dataclass

from thua.trace.client import StreamResult
from thua.trace.loader import Request


@dataclass
class ReplayItem:
    request: Request
    result: StreamResult


class Replayer:
    def __init__(self, client, clock=time.perf_counter, sleep=asyncio.sleep, time_scale=1.0):
        self.client = client
        self.clock = clock
        self.sleep = sleep
        self.time_scale = time_scale

    async def run(self, requests: list[Request]) -> list[ReplayItem]:
        results: list[StreamResult | None] = [None] * len(requests)

        async def _one(idx: int, req: Request):
            delay = (req.timestamp_ms / 1000.0) * self.time_scale
            if delay > 0:
                await self.sleep(delay)
            results[idx] = await self.client.stream(req.body)

        await asyncio.gather(*(_one(i, r) for i, r in enumerate(requests)))
        return [ReplayItem(req, res) for req, res in zip(requests, results)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trace/test_replayer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/trace/replayer.py tests/trace/test_replayer.py
git commit -m "feat: async trace replayer honoring request timestamps"
```

---

### Task 8: Speed evaluator + report

**Files:**
- Create: `src/thua/evaluation/__init__.py` (empty)
- Create: `src/thua/evaluation/base.py`
- Create: `src/thua/evaluation/speed.py`
- Create: `src/thua/evaluation/report.py`
- Test: `tests/evaluation/test_speed.py`

**Interfaces:**
- Consumes: `Replayer`, `ReplayItem` (Task 7); `compute_timing` (Task 3); `ScoringConfig`, `request_score`, `ers` (Task 1).
- Produces: `class Evaluator(ABC)` with `async def evaluate(self) -> dict` in `base.py`; `class SpeedEvaluator(Evaluator)` with `__init__(self, requests, replayer, scoring_cfg)` and `async def evaluate(self) -> dict` returning `{"ers": float, "n": int, "n_success": int, "per_request": [{"request_id", "ttft_ms", "tpot_ms", "score", "success"}], "ttft_mean_ms", "tpot_mean_ms"}`. In `report.py`: `format_speed_report(result: dict) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_speed.py
import pytest
from thua.trace.loader import Request
from thua.trace.client import StreamResult
from thua.trace.replayer import Replayer
from thua.metrics.scoring import ScoringConfig
from thua.evaluation.speed import SpeedEvaluator

class FakeClient:
    def __init__(self, results):
        self._results = results
        self._i = 0
    async def stream(self, body):
        r = self._results[self._i]
        self._i += 1
        return r

@pytest.mark.asyncio
async def test_speed_evaluator_computes_ers():
    reqs = [
        Request(0, 0, {"messages": [{"role": "user", "content": "a"}]}),
        Request(1, 0, {"messages": [{"role": "user", "content": "b"}]}),
    ]
    # req0: ttft=100ms, then tokens at +20ms each -> best case; req1: failure
    results = [
        StreamResult(token_times_ms=[100.0, 120.0, 140.0], text="x", success=True),
        StreamResult(token_times_ms=[], text="", success=False, error="timeout"),
    ]
    ev = SpeedEvaluator(reqs, Replayer(FakeClient(results), time_scale=0.0), ScoringConfig())
    out = await ev.evaluate()
    assert out["n"] == 2
    assert out["n_success"] == 1
    # req0 score ~1.0, req1 score 0 -> ERS ~0.5
    assert abs(out["ers"] - 0.5) < 1e-6
    assert out["per_request"][1]["success"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_speed.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `base.py`**

```python
# src/thua/evaluation/base.py
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self) -> dict:
        ...
```

- [ ] **Step 4: Write `speed.py`**

```python
# src/thua/evaluation/speed.py
from statistics import mean

from thua.evaluation.base import Evaluator
from thua.metrics.latency import compute_timing
from thua.metrics.scoring import ScoringConfig, request_score, ers


class SpeedEvaluator(Evaluator):
    def __init__(self, requests, replayer, scoring_cfg: ScoringConfig):
        self.requests = requests
        self.replayer = replayer
        self.scoring_cfg = scoring_cfg

    async def evaluate(self) -> dict:
        items = await self.replayer.run(self.requests)
        per_request = []
        scores = []
        ttfts = []
        tpots = []
        for item in items:
            t = compute_timing(item.result.token_times_ms, item.result.success)
            score = request_score(t.ttft_ms, t.tpot_ms, t.success, self.scoring_cfg)
            scores.append(score)
            if t.success:
                if t.ttft_ms is not None:
                    ttfts.append(t.ttft_ms)
                if t.tpot_ms is not None:
                    tpots.append(t.tpot_ms)
            per_request.append({
                "request_id": item.request.request_id,
                "ttft_ms": t.ttft_ms,
                "tpot_ms": t.tpot_ms,
                "score": score,
                "success": t.success,
            })
        return {
            "ers": ers(scores),
            "n": len(items),
            "n_success": sum(1 for p in per_request if p["success"]),
            "per_request": per_request,
            "ttft_mean_ms": mean(ttfts) if ttfts else None,
            "tpot_mean_ms": mean(tpots) if tpots else None,
        }
```

- [ ] **Step 5: Write `report.py`**

```python
# src/thua/evaluation/report.py
def format_speed_report(result: dict) -> str:
    lines = [
        "=== Speed ===",
        f"ERS:          {result['ers']:.4f}",
        f"Requests:     {result['n_success']}/{result['n']} succeeded",
    ]
    if result.get("ttft_mean_ms") is not None:
        lines.append(f"TTFT mean:    {result['ttft_mean_ms']:.1f} ms")
    if result.get("tpot_mean_ms") is not None:
        lines.append(f"TPOT mean:    {result['tpot_mean_ms']:.1f} ms")
    return "\n".join(lines)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/evaluation/test_speed.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Commit**

```bash
git add src/thua/evaluation/ tests/evaluation/test_speed.py
git commit -m "feat: speed evaluator (replay -> ERS) + report formatter"
```

---

### Task 9: GPQA accuracy evaluator

**Files:**
- Create: `src/thua/evaluation/gpqa.py`
- Test: `tests/evaluation/test_gpqa.py`

**Interfaces:**
- Consumes: `delta` (Task 2).
- Produces: `format_question(row: dict, rng) -> tuple[str, str]` (returns `(prompt, gold_letter)`); `extract_answer(text: str) -> str | None`; `class GpqaEvaluator` with `__init__(self, rows, chat_fn, baseline=0.4)` and `async def evaluate(self) -> dict` returning `{"accuracy": float, "n": int, "n_correct": int, "delta": float}`. `chat_fn` is `async (prompt: str) -> str` (injected so tests need no server and no dataset download). `rows` is an iterable of dicts with keys `Question`, `Correct Answer`, `Incorrect Answer 1..3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_gpqa.py
import random
import pytest
from thua.evaluation.gpqa import GpqaEvaluator, extract_answer, format_question

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_gpqa.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/evaluation/gpqa.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_gpqa.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/evaluation/gpqa.py tests/evaluation/test_gpqa.py
git commit -m "feat: GPQA Diamond accuracy evaluator (injectable dataset + chat_fn)"
```

---

### Task 10: Serve backend + docker-compose generation

**Files:**
- Create: `src/thua/serve/__init__.py` (empty)
- Create: `src/thua/serve/base.py`
- Create: `src/thua/serve/vllm.py`
- Create: `src/thua/serve/compose.py`
- Create: `src/thua/serve/registry.py`
- Test: `tests/serve/test_vllm.py`

**Interfaces:**
- Consumes: `ServeConfig`, `KVCacheConfig`, `SchedulingConfig`, `QuantConfig` (Task 4).
- Produces: `class ServeBackend(ABC)` with `build_args(self) -> list[str]`, `compose_service(self) -> dict`, `health_url(self) -> str`; `class VllmBackend(ServeBackend)` with `__init__(self, serve, kvcache, scheduling, quant)`; `get_backend(name: str, ...) -> ServeBackend` in `registry.py`; `to_compose_yaml(backend: ServeBackend, image: str) -> str` in `compose.py`.
- The first 4 args MUST be exactly `--model=<path>`, `--served-model-name=<name>`, `--host=<host>`, `--port=<port>` in that order (competition rule).

- [ ] **Step 1: Write the failing test**

```python
# tests/serve/test_vllm.py
import yaml
from thua.config.schema import ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig
from thua.serve.vllm import VllmBackend
from thua.serve.compose import to_compose_yaml

def _backend():
    return VllmBackend(
        ServeConfig(model_path="/model", served_model_name="Qwen3.5-2B",
                    host="0.0.0.0", port=8000, max_model_len=45000,
                    gpu_memory_utilization=0.9),
        KVCacheConfig(enable_prefix_caching=True, kv_cache_dtype="fp8_e4m3"),
        SchedulingConfig(enable_chunked_prefill=True, max_num_batched_tokens=8192,
                         max_num_seqs=32),
        QuantConfig(mode="online", method="fp8"),
    )

def test_first_four_args_fixed_and_ordered():
    args = _backend().build_args()
    assert args[:4] == [
        "--model=/model", "--served-model-name=Qwen3.5-2B",
        "--host=0.0.0.0", "--port=8000",
    ]

def test_kvcache_and_scheduling_args_present():
    args = _backend().build_args()
    assert "--enable-prefix-caching" in args
    assert "--kv-cache-dtype=fp8_e4m3" in args
    assert "--max-num-batched-tokens=8192" in args
    assert "--quantization=fp8" in args
    assert "--max-model-len=45000" in args

def test_compose_yaml_is_valid_and_keeps_entrypoint():
    doc = yaml.safe_load(to_compose_yaml(_backend(), image="me/thua:latest"))
    svc = doc["services"]["model"]
    assert svc["image"] == "me/thua:latest"
    assert svc["entrypoint"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert svc["command"][:4] == [
        "--model=/model", "--served-model-name=Qwen3.5-2B",
        "--host=0.0.0.0", "--port=8000",
    ]
    assert "healthcheck" in svc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/serve/test_vllm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `base.py`**

```python
# src/thua/serve/base.py
from abc import ABC, abstractmethod


class ServeBackend(ABC):
    @abstractmethod
    def build_args(self) -> list[str]:
        ...

    @abstractmethod
    def compose_service(self) -> dict:
        ...

    @abstractmethod
    def health_url(self) -> str:
        ...
```

- [ ] **Step 4: Write `vllm.py`**

```python
# src/thua/serve/vllm.py
from thua.config.schema import ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig
from thua.serve.base import ServeBackend

ENTRYPOINT = ["python3", "-m", "vllm.entrypoints.openai.api_server"]


class VllmBackend(ServeBackend):
    def __init__(self, serve: ServeConfig, kvcache: KVCacheConfig,
                 scheduling: SchedulingConfig, quant: QuantConfig):
        self.serve = serve
        self.kvcache = kvcache
        self.scheduling = scheduling
        self.quant = quant

    def build_args(self) -> list[str]:
        s = self.serve
        args = [
            f"--model={s.model_path}",
            f"--served-model-name={s.served_model_name}",
            f"--host={s.host}",
            f"--port={s.port}",
            f"--max-model-len={s.max_model_len}",
            f"--gpu-memory-utilization={s.gpu_memory_utilization}",
            "--tensor-parallel-size=1",
        ]
        args += self._kvcache_args()
        args += self._scheduling_args()
        args += self._quant_args()
        for k, v in s.extra_args.items():
            args.append(f"--{k}={v}" if v is not None else f"--{k}")
        return args

    def _kvcache_args(self) -> list[str]:
        kv = self.kvcache
        out = []
        if kv.enable_prefix_caching:
            out.append("--enable-prefix-caching")
        if kv.kv_cache_dtype and kv.kv_cache_dtype != "auto":
            out.append(f"--kv-cache-dtype={kv.kv_cache_dtype}")
        out.append(f"--block-size={kv.block_size}")
        if kv.cpu_offload_gb and kv.cpu_offload_gb > 0:
            out.append(f"--cpu-offload-gb={kv.cpu_offload_gb}")
        return out

    def _scheduling_args(self) -> list[str]:
        sc = self.scheduling
        out = []
        if sc.enable_chunked_prefill:
            out.append("--enable-chunked-prefill")
        out.append(f"--max-num-batched-tokens={sc.max_num_batched_tokens}")
        out.append(f"--max-num-seqs={sc.max_num_seqs}")
        if sc.long_prefill_token_threshold > 0:
            out.append(f"--long-prefill-token-threshold={sc.long_prefill_token_threshold}")
        if sc.max_num_partial_prefills > 1:
            out.append(f"--max-num-partial-prefills={sc.max_num_partial_prefills}")
        out.append(f"--scheduling-policy={sc.scheduling_policy}")
        out.append(f"--preemption-mode={sc.preemption_mode}")
        return out

    def _quant_args(self) -> list[str]:
        if self.quant.mode == "online" and self.quant.method not in ("none", None):
            return [f"--quantization={self.quant.method}"]
        return []

    def compose_service(self) -> dict:
        return {
            "image": "REPLACED_BY_COMPOSE",
            "entrypoint": list(ENTRYPOINT),
            "command": self.build_args(),
            "ports": [f"{self.serve.port}:{self.serve.port}"],
            "shm_size": "2g",
            "healthcheck": {
                "test": ["CMD", "python3", "-c",
                         f"import urllib.request,sys; "
                         f"sys.exit(0) if urllib.request.urlopen('http://localhost:{self.serve.port}/health').status==200 else sys.exit(1)"],
                "interval": "10s",
                "timeout": "5s",
                "retries": 30,
                "start_period": "300s",
            },
            "deploy": {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}},
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.serve.port}/health"
```

- [ ] **Step 5: Write `compose.py` and `registry.py`**

```python
# src/thua/serve/compose.py
import yaml
from thua.serve.base import ServeBackend


def to_compose_yaml(backend: ServeBackend, image: str) -> str:
    svc = backend.compose_service()
    svc["image"] = image
    doc = {"services": {"model": svc}}
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
```

```python
# src/thua/serve/registry.py
from thua.config.schema import ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig
from thua.serve.base import ServeBackend
from thua.serve.vllm import VllmBackend

_BACKENDS = {"vllm": VllmBackend}


def get_backend(name: str, serve: ServeConfig, kvcache: KVCacheConfig,
                scheduling: SchedulingConfig, quant: QuantConfig) -> ServeBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend: {name} (have {list(_BACKENDS)})")
    return _BACKENDS[name](serve, kvcache, scheduling, quant)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/serve/test_vllm.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add src/thua/serve/ tests/serve/test_vllm.py
git commit -m "feat: pluggable vLLM serve backend + docker-compose generation"
```

---

### Task 11: Server stats scraper (Prometheus /metrics)

**Files:**
- Create: `src/thua/metrics/server_stats.py`
- Test: `tests/metrics/test_server_stats.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_metrics(text: str) -> dict[str, float]` — parses Prometheus exposition text into `{metric_name: value}` for the last sample of each series, ignoring `#` comment lines and labels. `KEY_METRICS: list[str]` — the subset we care about (prefix cache hit rate, running/waiting, cache usage, preemptions).

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_server_stats.py
from thua.metrics.server_stats import parse_metrics

SAMPLE = """# HELP vllm:gpu_prefix_cache_hit_rate ...
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="Qwen3.5-2B"} 0.82
vllm:num_requests_running{model_name="Qwen3.5-2B"} 12.0
vllm:num_requests_waiting{model_name="Qwen3.5-2B"} 8.0
"""

def test_parse_extracts_values():
    m = parse_metrics(SAMPLE)
    assert abs(m["vllm:gpu_prefix_cache_hit_rate"] - 0.82) < 1e-9
    assert m["vllm:num_requests_running"] == 12.0

def test_parse_ignores_comments():
    m = parse_metrics(SAMPLE)
    assert all(not k.startswith("#") for k in m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/metrics/test_server_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/thua/metrics/server_stats.py
KEY_METRICS = [
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
]


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        if not name_part:
            continue
        name = name_part.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/metrics/test_server_stats.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/thua/metrics/server_stats.py tests/metrics/test_server_stats.py
git commit -m "feat: Prometheus /metrics parser for KV/scheduling observability"
```

---

### Task 12: main.py CLI + scripts wiring

**Files:**
- Create: `src/thua/runner.py`
- Create: `main.py`
- Create: `scripts/run_speed.py`
- Create: `scripts/gen_compose.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `load_experiment` (Task 4), `load_trace` (Task 5), `StreamingClient` (Task 6), `Replayer` (Task 7), `SpeedEvaluator` (Task 8), `ScoringConfig` (Task 1), `format_speed_report` (Task 8), `final_score` (Task 2), `get_backend` + `to_compose_yaml` (Task 10).
- Produces: in `runner.py` `async def run_speed(cfg) -> dict` and `def gen_compose(cfg, image: str) -> str`; `main.py` argparse dispatch with subcommands `speed`, `compose`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
from thua.config.loader import load_experiment
from thua.runner import gen_compose

def _cfg(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "serve: {backend: vllm, model_path: /model, served_model_name: Qwen3.5-2B, "
        "host: 0.0.0.0, port: 8000, max_model_len: 45000, gpu_memory_utilization: 0.9, "
        "dtype: auto, extra_args: {}}\n"
        "kvcache: {enable_prefix_caching: true, kv_cache_dtype: fp8_e4m3, block_size: 16, "
        "cpu_offload_gb: 0}\n"
        "scheduling: {enable_chunked_prefill: true, max_num_batched_tokens: 8192, "
        "long_prefill_token_threshold: 0, max_num_seqs: 32, max_num_partial_prefills: 1, "
        "scheduling_policy: fcfs, preemption_mode: recompute}\n"
        "quant: {mode: online, method: fp8}\n"
        "benchmark: {trace_path: trace-round1.jsonl, base_url: http://localhost:8000/v1, "
        "model: Qwen3.5-2B, timeout_s: 60}\n"
        "accuracy: {dataset: Idavidrein/gpqa, subset: gpqa_diamond, n: 198, baseline: 0.4, "
        "base_url: http://localhost:8000/v1, model: Qwen3.5-2B, timeout_s: 120}\n",
        encoding="utf-8",
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text("name: t\n", encoding="utf-8")
    return load_experiment(str(exp), base_path=str(base))

def test_gen_compose_produces_image_and_command(tmp_path):
    out = gen_compose(_cfg(tmp_path), image="me/thua:latest")
    assert "me/thua:latest" in out
    assert "--kv-cache-dtype=fp8_e4m3" in out
    assert "--quantization=fp8" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `runner.py`**

```python
# src/thua/runner.py
from thua.config.schema import ExperimentConfig
from thua.evaluation.speed import SpeedEvaluator
from thua.metrics.scoring import ScoringConfig
from thua.serve.compose import to_compose_yaml
from thua.serve.registry import get_backend
from thua.trace.client import StreamingClient
from thua.trace.loader import load_trace
from thua.trace.replayer import Replayer


async def run_speed(cfg: ExperimentConfig) -> dict:
    requests = load_trace(cfg.benchmark.trace_path)
    client = StreamingClient(cfg.benchmark.base_url, cfg.benchmark.model,
                             cfg.benchmark.timeout_s)
    replayer = Replayer(client)
    evaluator = SpeedEvaluator(requests, replayer, ScoringConfig())
    return await evaluator.evaluate()


def gen_compose(cfg: ExperimentConfig, image: str) -> str:
    backend = get_backend(cfg.serve.backend, cfg.serve, cfg.kvcache,
                          cfg.scheduling, cfg.quant)
    return to_compose_yaml(backend, image)
```

- [ ] **Step 4: Write `main.py`**

```python
# main.py
import argparse
import asyncio

from thua.config.loader import load_experiment
from thua.evaluation.report import format_speed_report
from thua.runner import run_speed, gen_compose


def _parse():
    p = argparse.ArgumentParser(prog="thua")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("speed", help="replay trace and report ERS")
    sp.add_argument("--config", required=True)
    sp.add_argument("--base-config", default="configs/base.yaml")
    gc = sub.add_parser("compose", help="generate docker-compose.yml")
    gc.add_argument("--config", required=True)
    gc.add_argument("--base-config", default="configs/base.yaml")
    gc.add_argument("--image", required=True)
    gc.add_argument("--out", default="docker-compose.generated.yml")
    return p.parse_args()


def main():
    args = _parse()
    cfg = load_experiment(args.config, base_path=args.base_config)
    if args.cmd == "speed":
        result = asyncio.run(run_speed(cfg))
        print(format_speed_report(result))
    elif args.cmd == "compose":
        yaml_text = gen_compose(cfg, image=args.image)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `scripts/run_speed.py` and `scripts/gen_compose.py`**

```python
# scripts/run_speed.py
"""Thin wrapper: python scripts/run_speed.py --config configs/experiment/exp_fp8.yaml"""
import sys
from main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "speed"] + sys.argv[1:]
    main()
```

```python
# scripts/gen_compose.py
"""Thin wrapper: python scripts/gen_compose.py --config ... --image me/thua:latest"""
import sys
from main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "compose"] + sys.argv[1:]
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_runner.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Run the whole suite + generate a real compose**

Run: `pytest -q`
Expected: all tests pass.
Run: `python main.py compose --config configs/experiment/exp_fp8.yaml --image me/thua:latest --out docker-compose.generated.yml`
Expected: writes `docker-compose.generated.yml`; open it and confirm the first 4 command args are the fixed ones and `--kv-cache-dtype=fp8_e4m3` is present.

- [ ] **Step 8: Commit**

```bash
git add src/thua/runner.py main.py scripts/run_speed.py scripts/gen_compose.py tests/test_runner.py
git commit -m "feat: main CLI (speed + compose) and script wrappers"
```

---

## Notes for the implementer

- Add a `.gitignore` with `results/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/` when convenient (fold into Task 1 if you prefer).
- `datasets` is only needed for the *real* GPQA run (loading `Idavidrein/gpqa`); the tests inject rows, so you can implement and test everything here without it. The real dataset wiring (a `load_gpqa_rows(cfg)` helper feeding `GpqaEvaluator`) belongs in Plan 2 alongside the GPU/serving pieces, or add it as a follow-up task once you have HF access.
- Speed replay against a real server: `python main.py speed --config configs/experiment/exp_fp8.yaml` after `docker compose up` on the L4/MiG box.

## Deferred to Plan 2 (needs GPU / heavy deps)

- `src/thua/model/` offline quantization (llm-compressor FP8/AWQ/GPTQ) + `configs/quantize/`.
- `docker/Dockerfile` self-contained image (bake quantized `/model`).
- `scripts/build_and_push.sh`, `scripts/sweep.py` (grid-search KV × scheduling), real GPQA dataset wiring, live `server_stats` scraping during a run.
