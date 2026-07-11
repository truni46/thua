# THUA — Qwen3.5-2B Serving & Benchmark Pipeline — Design

**Date:** 2026-07-08
**Status:** Approved (pending spec review)

## 1. Context & Goal

Competition Phase 1: serve `Qwen/Qwen3.5-2B` (dense, BF16 origin) and maximize the
Effective Request Score (ERS) over a 120-request trace while passing the GPQA Diamond
accuracy gate. BTC evaluates automatically on **1× MiG H200 (18GB VRAM, 3 CPU, 8GB RAM)**:
they pull a public Docker image, bring up the container from a submitted
`docker-compose.yml`, run a healthcheck, then benchmark.

Development happens on an **L4 (24GB)** — a valid proxy: same VRAM class, Ada supports FP8
like Hopper, comparable compute. L4 memory bandwidth is ~2× lower than the MiG slice, so
measured TTFT/TPOT on L4 are **pessimistic** (pass on L4 ⇒ pass on MiG). Absolute latency
numbers do not transfer 1:1.

### Workload characterization (from `trace-round1.jsonl`)

- 120 requests, all `conversation`, 2 messages (system+user), `temperature=0`, `seed=42`.
- **Long-context, prefill-bound:** prompts median ~30k tokens (max ~42k); `max_tokens=200`.
- **Shared prefix across ALL 120 requests: ~9,700 tokens** (38,969 chars); some clusters
  share up to ~131k chars.
- Arrivals span 25.5s; **~20 requests burst at t=0**, rest trickle (driven by `timestamp_ms`).

**Consequence:** TTFT (prefill of ~30k tokens) is the battleground; TPOT (decode of 200
tokens on a 2B model) is easy. The score is won by **KV-cache reuse (prefix caching)** and
**scheduling of the concurrent long prefills** — quantization is secondary (near-free).

### Scoring (from đề)

- `ERS = mean(S_request)`, `S_request = w·s_ttft + (1-w)·s_tpot` on success, else `0`
  (error / timeout / 0 tokens → 0).
- `s_ttft = clamp((C_ttft - TTFT)/(C_ttft - F_ttft), 0, 1)^γ`,
  `s_tpot = clamp((C_tpot - TPOT_mean)/(C_tpot - F_tpot), 0, 1)^γ`.
- Constants: `F_ttft=100ms, C_ttft=1500ms, F_tpot=20ms, C_tpot=45ms, γ=2, w=0.5`.
- Accuracy: `Δ = baseline(0.4) - acc`; `f(Δ)=1 if Δ≤0.1`, linear down to `0` at `Δ≥0.16`.
- `Score = 100 · ERS · f(Δ)`. → ~10pt accuracy headroom ⇒ FP8 essentially free.

## 2. Scope

Full solution (scope **C**): evaluation harness + server config management + submission
packaging. **Dev environment has no GPU yet** → pipeline is scaffold-first: it *generates*
`docker-compose.yml` / `Dockerfile` and benchmarks against a `base_url` endpoint. Auto-launch
(`docker compose up`) is behind a clean interface to add later on L4/MiG.

Serving runtime is **pluggable** (vLLM primary, SGLang secondary) via a `ServeBackend` ABC.

## 3. Architecture

### Design principle → mapping from user's DL-training mental model

| DL training | This pipeline | Note |
|---|---|---|
| `config/` | `configs/` | YAML |
| `model/` | `model/` = **quantize** model | no training; still "prepare model" |
| `trainer/` | `evaluation/` = **Runners** | run model over data → metrics |
| `metric.py` | `metrics/` | ERS + f(Δ) + server stats |
| `scripts/` | `scripts/` | one-off tooling |
| `main.py` | `main.py` | orchestrate per experiment config |
| — | `serve/`, `trace/` | serving-specific additions |

### Folder structure

```
THUA/
├── main.py                     # python main.py --config configs/experiment/exp_fp8.yaml
├── configs/
│   ├── base.yaml
│   ├── serve/       vllm_fp8.yaml · vllm_bf16.yaml · sglang_fp8.yaml
│   ├── kvcache/     prefix_on.yaml · fp8_kv.yaml · offload.yaml · semantic.yaml(opt,off)
│   ├── scheduling/  chunked.yaml · batching.yaml · policy.yaml · spec.yaml(off)
│   ├── quantize/    fp8_w8a8.yaml · awq_int4.yaml · gptq_int4.yaml
│   ├── benchmark/   trace_round1.yaml       # trace path, timeline, ERS constants
│   ├── accuracy/    gpqa_diamond.yaml       # dataset, n, prompt template, baseline
│   └── experiment/  exp_fp8.yaml            # composes serve+kvcache+scheduling+quant+bench+acc
├── src/thua/
│   ├── config/     schema.py (typed dataclasses) · loader.py (base+override merge)
│   ├── model/      base.py(Quantizer ABC) · fp8.py · awq.py · gptq.py · online.py(no-op) · spec.py · registry.py
│   ├── serve/      base.py(ServeBackend ABC) · vllm.py · sglang.py · compose.py · registry.py
│   ├── trace/      loader.py(jsonl→Request) · client.py(async streaming) · replayer.py(honor timestamp_ms)
│   ├── metrics/    latency.py(TTFT/TPOT) · scoring.py(ERS) · accuracy.py(Δ,f(Δ),Score) · server_stats.py(/metrics)
│   ├── evaluation/ base.py(Evaluator ABC) · speed.py · gpqa.py · report.py
│   └── utils/      io.py · logging.py
├── scripts/        quantize_model.py · gen_compose.py · run_speed.py · run_gpqa.py · sweep.py · build_and_push.sh
├── docker/         Dockerfile
├── results/        (gitignored)
└── pyproject.toml
```

### Key abstractions (OOP, pluggable)

```python
class ServeBackend(ABC):                 # VllmBackend, SglangBackend
    def kvcache_args(self, cfg: KVCacheConfig) -> list[str]: ...
    def scheduling_args(self, cfg: SchedulingConfig) -> list[str]: ...
    def quant_args(self, cfg: QuantConfig) -> list[str]: ...
    def build_args(self) -> list[str]: ...          # merges all; keeps required entrypoint args fixed
    def compose_service(self) -> dict: ...          # docker-compose service incl. healthcheck
    def health_url(self) -> str: ...

class Quantizer(ABC):                    # Fp8Quantizer, AwqQuantizer, GptqQuantizer, OnlineQuantizer(no-op)
    def run(self, src_model: str, out_dir: str) -> str: ...   # returns /model path

class Evaluator(ABC):                    # SpeedEvaluator, GpqaEvaluator
    def evaluate(self, endpoint: str) -> Result: ...
```

`KVCacheConfig` and `SchedulingConfig` are standalone dataclasses shared across backends;
each backend maps them to its own flag names (vLLM vs SGLang differ).

## 4. Optimization coverage (đề's Optimization Scope → where it lives → priority)

- **Quantization** (FP8/INT8/INT4/AWQ/GPTQ/activation): `model/` offline + online serve flag
  + `configs/quantize/`. Priority: secondary (near-free on accuracy).
- **KV Cache**: paged attention (vLLM default, baseline); KV quant FP8/INT8 (`kvcache/fp8_kv`);
  **prefix caching** (`kvcache/prefix_on`) — **highest priority**; semantic caching
  (`kvcache/semantic`, off by default — accuracy/gaming risk); CPU/NVMe offload
  (`kvcache/offload`, fallback). Priority: **primary**.
- **Serving & Scheduling**: continuous batching (`scheduling/batching`); chunked prefill
  (`scheduling/chunked`) — **primary** for the t=0 burst; speculative decoding
  (`model/spec` + `scheduling/spec`, off by default — prefill-bound). Priority: **primary**.
- **System & Runtime**: FlashInfer/FlashAttention (serve env + config) — primary for
  long-context; CUDA graphs (no `enforce_eager`) baseline; custom CUDA/Triton kernels —
  YAGNI (interfaces left in place).

**Scheduling is server-side** (vLLM scheduler config), not client reordering — BTC sends the
trace; the replayer only reproduces it faithfully. A candidate lever for the burst:
throttle concurrent prefills (`max_num_partial_prefills`) so request 1 populates the prefix
cache before the other ~19 hit it, avoiding redundant recompute of the 9.7k shared tokens.

## 5. Observability (closes the tuning loop)

`metrics/server_stats.py` scrapes vLLM Prometheus `/metrics` and attaches to each speed run:
`gpu_prefix_cache_hit_rate`, `num_requests_running`/`waiting`, `gpu_cache_usage_perc`,
preemption counts. Lets us see *why* a config is fast/slow, not just the ERS number.

## 6. Data flow

```
main.py(exp.yaml)
  → config.loader → typed configs
  → (optional) model.Quantizer → /model checkpoint     # offline FP8/AWQ
  → serve.compose.gen → docker-compose.yml + Dockerfile # scaffold; no GPU locally
  → assume endpoint at base_url                          # docker compose up done manually on L4/MiG
  → SpeedEvaluator: trace.replayer replays 120 req (async, honoring timestamp_ms, streaming)
        → metrics.latency → metrics.scoring → ERS  (+ server_stats)
  → GpqaEvaluator: run GPQA Diamond (198 q) → metrics.accuracy → Δ, f(Δ)
  → evaluation.report: table + JSON, Score = 100·ERS·f(Δ)
```

## 7. Submission packaging (workflow: build → push public → submit compose → BTC pulls)

- `docker/Dockerfile`: **self-contained** — bake offline-quantized weights into `/model`
  (no HF download at runtime → avoids network slowness/failure and healthcheck timeout).
- `serve/compose.py`: emits a valid `docker-compose.yml` with a `/health` healthcheck; keeps
  the required entrypoint + first 4 args fixed (competition rule); only optimized args vary.
- Startup budget: use `max_model_len≈45000` (not 262144) and tune `cudagraph_capture_sizes`
  so model load + CUDA graph capture finishes before the healthcheck deadline.
- `scripts/build_and_push.sh`: build + `docker push` to public Docker Hub.
- `scripts/gen_compose.py`: produces the final submission `docker-compose.yml`.

## 8. Decisions locked

- Config: plain YAML + dataclasses, manual `base + override` merge (no Hydra — YAGNI).
- Async replayer: `asyncio + httpx`, streaming, to reproduce the t=0 burst and measure
  first-token TTFT accurately.
- Scaffold mode default (no GPU); auto-launch behind interface for later.
- Speculative decoding & semantic caching implemented but **off by default**.

## 9. Accuracy-safety procedure

Measure `acc_bf16` vs `acc_fp8` (and `acc_int4` if pushed) on the **same** GPQA protocol;
keep `Δ ≤ 0.1` (target margin). FP8 expected `Δ < 0.03` (baseline 0.4 is near-random) → safe;
INT4/AWQ must be re-measured before use.

## 10. Out of scope (YAGNI)

Custom CUDA/Triton kernels; TensorRT-LLM backend; multi-GPU/TP>1; training/fine-tuning;
Hydra; automatic client-side request reordering.
```
